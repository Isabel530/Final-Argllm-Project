"""Model loading and score generation with explicit generation diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Literal

from .constraints import make_prefix_allowed_tokens_fn
from .prompts import PromptSpec


@dataclass(frozen=True)
class GenerationSettings:
    decoding: str = "hard"
    max_new_tokens: int | Literal["context_window"] = 12
    do_sample: bool = True
    temperature: float = 0.7
    top_p: float = 0.95
    repetition_penalty: float = 1.0
    thinking: bool = False

    def __post_init__(self) -> None:
        if self.decoding not in {"hard", "soft"}:
            raise ValueError("decoding must be 'hard' or 'soft'")
        if self.max_new_tokens != "context_window" and (
            type(self.max_new_tokens) is not int or self.max_new_tokens <= 0
        ):
            raise ValueError(
                "max_new_tokens must be a positive integer or 'context_window'"
            )
        if type(self.thinking) is not bool:
            raise TypeError("thinking must be boolean")


@dataclass(frozen=True)
class GenerationOutput:
    raw_output: str
    elapsed_seconds: float
    generated_tokens: int
    hit_max_new_tokens: bool
    max_new_tokens_limit: int


class HuggingFaceScoreGenerator:
    """One loaded Hugging Face causal LM, reused across prompt conditions."""

    def __init__(self, model_name: str, **model_config: Any) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError as exc:
            raise RuntimeError(
                "Generation requires torch and transformers; install requirements.txt"
            ) from exc

        self._torch = torch
        self.model_name = model_name
        self.revision = model_config.get("revision")
        cache_dir = model_config.get("cache_dir")
        trust_remote_code = bool(model_config.get("trust_remote_code", True))
        dtype_name = model_config.get("torch_dtype", "bfloat16")
        dtype = getattr(torch, dtype_name)
        quantization = model_config.get("quantization", "none")

        quantization_config = None
        if quantization == "4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=dtype
            )
        elif quantization == "8bit":
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        elif quantization != "none":
            raise ValueError(f"Unsupported quantization: {quantization}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            revision=self.revision,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code,
        )
        load_kwargs: dict[str, Any] = {
            "revision": self.revision,
            "cache_dir": cache_dir,
            "trust_remote_code": trust_remote_code,
            "device_map": model_config.get("device_map", "auto"),
            "torch_dtype": dtype,
        }
        if quantization_config is not None:
            load_kwargs["quantization_config"] = quantization_config
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        self.model.eval()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def _chat_prompt(self, message: str, thinking: bool) -> str:
        if "gpt-oss" in self.model_name.lower():
            message = f"Reasoning: {'high' if thinking else 'low'}\n\n{message}"
        messages = [{"role": "user", "content": message}]
        kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
        if "qwen" in self.model_name.lower():
            kwargs["enable_thinking"] = thinking
        return self.tokenizer.apply_chat_template(messages, **kwargs)

    def _effective_max_new_tokens(
        self,
        configured: int | Literal["context_window"],
        *,
        prompt_length: int,
    ) -> int:
        if configured != "context_window":
            return configured

        model_config = self.model.config
        context_window = getattr(model_config, "max_position_embeddings", None)
        if context_window is None and hasattr(model_config, "text_config"):
            context_window = getattr(
                model_config.text_config, "max_position_embeddings", None
            )
        if type(context_window) is not int or context_window <= 0:
            raise ValueError(
                f"{self.model_name} does not expose a valid context-window size"
            )

        available = context_window - prompt_length
        if available <= 0:
            raise ValueError(
                f"Prompt length {prompt_length} leaves no generation space in "
                f"{context_window}-token context window"
            )
        return available

    def generate(
        self,
        prompt: PromptSpec,
        *,
        seed: int,
        settings: GenerationSettings,
    ) -> GenerationOutput:
        from transformers import set_seed

        set_seed(seed)
        rendered = self._chat_prompt(prompt.text, settings.thinking)
        inputs = self.tokenizer(rendered, return_tensors="pt")
        input_device = next(self.model.parameters()).device
        inputs = {key: value.to(input_device) for key, value in inputs.items()}
        prompt_length = int(inputs["input_ids"].shape[-1])
        max_new_tokens = self._effective_max_new_tokens(
            settings.max_new_tokens,
            prompt_length=prompt_length,
        )

        generate_kwargs: dict[str, Any] = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "do_sample": settings.do_sample,
            "repetition_penalty": settings.repetition_penalty,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if settings.do_sample:
            generate_kwargs.update(temperature=settings.temperature, top_p=settings.top_p)
        if settings.decoding == "hard":
            generate_kwargs["prefix_allowed_tokens_fn"] = make_prefix_allowed_tokens_fn(
                tokenizer=self.tokenizer,
                prompt_length=prompt_length,
                allowed_outputs=prompt.allowed_outputs,
            )

        started = perf_counter()
        with self._torch.inference_mode():
            output_ids = self.model.generate(**generate_kwargs)
        elapsed = perf_counter() - started
        generated_ids = output_ids[0, prompt_length:]
        generated_tokens = int(generated_ids.shape[-1])
        eos = self.tokenizer.eos_token_id
        last_token = int(generated_ids[-1]) if generated_tokens else None
        hit_limit = generated_tokens >= max_new_tokens and last_token != eos
        raw = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        return GenerationOutput(
            raw_output=raw,
            elapsed_seconds=elapsed,
            generated_tokens=generated_tokens,
            hit_max_new_tokens=hit_limit,
            max_new_tokens_limit=max_new_tokens,
        )

    def close(self) -> None:
        import gc

        del self.model
        gc.collect()
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()
