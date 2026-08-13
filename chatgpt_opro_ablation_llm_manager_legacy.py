from abc import ABC, abstractmethod
import torch
import transformers
from transformers import AutoTokenizer, BitsAndBytesConfig
from transformers.generation import GenerationConfig
from utils import construct_constraint_fun
from openai import OpenAI
import os
import re

# 1. 终极提取器 (已修复 1.0 不乘 100 的 Bug)
def extract_correct_score(raw_text, prompt_type):
    text_to_search = raw_text
    if "</think>" in text_to_search:
        text_to_search = text_to_search.split("</think>")[-1]
    if "assistantfinal" in text_to_search:
        text_to_search = text_to_search.split("assistantfinal")[-1]

    # "chatgpt_ablation" must be routed the same way as "chatgpt" -- prefix
    # match instead of equality, otherwise it silently fell through to the
    # opro branch (wrong regex, wrong 0-100 rescaling).
    is_chatgpt = prompt_type.startswith("chatgpt")

    number_str = None
    if is_chatgpt:
        match = re.search(r'certainty(?: score)?[*\s:]*(\d+\.?\d*)', text_to_search, re.IGNORECASE)
        if match: number_str = match.group(1)
    else: # opro / opro_ablation
        match = re.search(r'(?:total|overall|final)[\s*]*quality[\s*]*score[*\s:=]*(\d+\.?\d*)', text_to_search, re.IGNORECASE)
        if not match:
            match = re.search(r'quality[\s*]*score[*\s:=]*(\d+\.?\d*)', text_to_search, re.IGNORECASE)
        if not match:
            match = re.search(r'score[*\s:=]*(\d+\.?\d*)', text_to_search, re.IGNORECASE)
        if match: 
            number_str = match.group(1)

    if not number_str and "assistantfinal" in raw_text.lower():
        post_marker_text = raw_text.split("assistantfinal")[-1]
        anchor_match = re.search(r'(\d+\.?\d*)', post_marker_text)
        if anchor_match:
            number_str = anchor_match.group(1)

    if not number_str:
        fraction_match = re.search(r'(\d+\.?\d*)\s*/\s*(?:100|10)', text_to_search)
        if fraction_match:
            number_str = fraction_match.group(1)
        else:
            numbers = re.findall(r'\d+\.\d+|\d+', text_to_search)
            number_str = numbers[-1] if numbers else "50"

    try:
        val = float(number_str)
    except ValueError:
        val = 50.0

    if is_chatgpt:
        if val > 1.0:
            val = val / 100.0
        return float(val)
    else:  # opro / opro_ablation
        if number_str is not None:
            local_context_match = re.search(rf'{re.escape(number_str)}\s*/\s*10(?!\d)', text_to_search)
            if local_context_match:
                val = val * 10
        if val <= 1.0 and val > 0 and "%" not in text_to_search:
            val = val * 100
        return int(val)


class LlmManager(ABC):
    @abstractmethod
    def chat_completion(self, prompt, print_result=False, seed=42, max_new_tokens=128, do_sample=True, temperature=0.7, top_p=0.95, repetition_penalty=1.0):
        pass


class HuggingFaceLlmManager(LlmManager):
    def __init__(self, model_name, cache_dir="argumentative-llm/cache", model_args=None, input_device="cuda:0", quantization="4bit"):
        super().__init__()
        if quantization == "4bit":
            quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
        elif quantization == "8bit":
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        elif quantization == "none":
            quantization_config = None
        else:
            raise ValueError(f"Invalid quantization value {quantization}")

        base_kwargs = {"torch_dtype": torch.bfloat16, "cache_dir": cache_dir}
        if quantization_config is not None:
            base_kwargs["quantization_config"] = quantization_config

        self.pipeline = transformers.pipeline(
            "text-generation", model=model_name, device_map="auto", model_kwargs=base_kwargs, trust_remote_code=True
        )
        self.input_device = input_device

    def chat_completion(self, message, print_result=False, seed=42, max_new_tokens=128, do_sample=True, temperature=0.7, top_p=0.95, repetition_penalty=1.0, constraint_prefix=None, constraint_options=None, constraint_end_after_options=False, trim_response=True, apply_template=True, enable_thinking=None):
        transformers.set_seed(seed)

        # 匹配单轮 Prompt 结构
        messages = [{"role": "user", "content": message}]

        if apply_template:
            template_kwargs = {}
            if enable_thinking is not None:
                template_kwargs["enable_thinking"] = enable_thinking
            prompt = self.pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **template_kwargs)
        else:
            prompt = message
            
        if constraint_prefix is not None or constraint_options is not None:
            prefix_allowed_tokens_fn = construct_constraint_fun(
                self.pipeline.tokenizer, prompt, force_prefix=constraint_prefix, force_options=constraint_options, end_after_options=constraint_end_after_options,
            )
        else:
            prefix_allowed_tokens_fn = None
            
        response = self.pipeline(
            prompt, max_new_tokens=max_new_tokens, do_sample=do_sample, temperature=temperature, top_p=top_p, repetition_penalty=repetition_penalty, prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        )[0]["generated_text"]

        if print_result:
            print(response, flush=True)

        if trim_response:
            response = response.replace(prompt, "").strip()

        return response


class CleanHuggingFaceManager(HuggingFaceLlmManager):
    def __init__(self, *args, **kwargs):
        self.model_name_internal = kwargs.get('model_name', args[0] if args else "")
        super().__init__(*args, **kwargs)

    def chat_completion(self, message, print_result=False, trim_response=True, **kwargs):
        is_think_mode = kwargs.pop("is_think_mode", False)
        prompt_type = kwargs.pop("prompt_type", "opro")
        
        # ⭐ 核心逻辑：精准读取主代码传进来的 use_constraints 指令
        use_constraints = kwargs.pop("use_constraints", False)
        constraint_prefix = kwargs.pop("constraint_prefix", None)
        constraint_options = kwargs.pop("constraint_options", None)
        end_after_options = kwargs.pop("constraint_end_after_options", False)
        
        # 如果需要开约束，且外面没提供选项，则在这里智能补齐
        # (In this experiment chatgpt_opro_ablation_prompt.py's prompt
        # functions already return concrete constraint_prefix/options, so
        # this fallback rarely triggers -- kept correct in case it ever does.)
        if use_constraints:
            if constraint_options is None:
                if prompt_type.startswith("chatgpt"):
                    constraint_prefix = "Certainty Score:"
                    constraint_options = [f" {x / 100}" for x in range(0, 101)]
                    end_after_options = True
                else: # opro / opro_ablation
                    constraint_prefix = "Quality score:"
                    constraint_options = [f" {x}" for x in range(0, 101)]
                    end_after_options = True
        else:
            constraint_prefix = None
            constraint_options = None
            end_after_options = False

        # gpt-oss 用 "Reasoning: low/high" 前缀控制推理强度（没有 chat-template 级别的
        # enable_thinking 开关），No-Think 时用 low。
        if "gpt-oss" in self.model_name_internal.lower():
            reasoning_directive = "Reasoning: high\n\n" if is_think_mode else "Reasoning: low\n\n"
            message = reasoning_directive + message

        # Qwen3 的思考模式开关（通过 chat template 的 enable_thinking 参数控制）
        if "qwen" in self.model_name_internal.lower():
            kwargs["enable_thinking"] = is_think_mode

        raw_response = super().chat_completion(
            message,
            print_result=False,
            trim_response=trim_response,
            constraint_prefix=constraint_prefix,
            constraint_options=constraint_options,
            constraint_end_after_options=end_after_options,
            **kwargs
        )

        final_val = extract_correct_score(raw_response, prompt_type)
        return str(final_val)


class OpenAiLlmManager(LlmManager):
    def __init__(self, model_name):
        self.model_name = model_name.split("openai/")[1]
        self.client = OpenAI(api_key=os.environ["OPENAI_KEY"])

    def chat_completion(self, message, print_result=False, seed=42, max_new_tokens=128, do_sample=True, temperature=0.7, top_p=0.95, repetition_penalty=1.0, constraint_prefix=None, constraint_options=None, constraint_end_after_options=False, trim_response=True, apply_template=True):
        api_messages = [{"role": "user", "content": message}]
        prompt = message

        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=api_messages,
            temperature=temperature,
            max_tokens=max_new_tokens,
            top_p=top_p,
            presence_penalty=repetition_penalty,
        )

        response = completion.choices[0].message.content

        if print_result:
            print(response, flush=True)

        if trim_response:
            response = response.replace(prompt, "").strip()

        return response