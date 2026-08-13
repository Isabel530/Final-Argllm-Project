"""Local language-model backend for critiques and readable prompt rewrites."""

from __future__ import annotations

import json
import re
from typing import Any, Sequence

from .apo import FailureCase, PromptCandidate, RewriteProposal


class OptimizerOutputError(ValueError):
    """Structured-output failure retaining raw attempts for audit/recovery."""

    def __init__(self, message: str, raw_outputs: Sequence[str]) -> None:
        super().__init__(message)
        self.raw_outputs = tuple(raw_outputs)


def _extract_json_object(text: str) -> dict[str, Any]:
    """Extract the proposal document, tolerating reasoning and code fences."""

    cleaned = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", text.strip())
    decoder = json.JSONDecoder()
    fallback = None
    for index, character in enumerate(cleaned):
        if character not in "{[":
            continue
        try:
            value, _ = decoder.raw_decode(cleaned[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, list):
            return {"proposals": value}
        if isinstance(value, dict):
            if isinstance(value.get("proposals"), list):
                return value
            fallback = value
    if fallback is not None:
        return fallback
    raise ValueError("optimizer output contains no valid JSON object")


def build_optimizer_request(
    parent: PromptCandidate,
    failures: Sequence[FailureCase],
    *,
    proposal_count: int,
) -> str:
    cases = "\n\n".join(
        f"Failure {index}:\n{failure.embedding_text()}\n"
        f"Seed margins (preferred minus other): {list(failure.seed_margins)}"
        for index, failure in enumerate(failures, start=1)
    ) or "No failure cases were available; improve clarity conservatively."
    return f"""You are optimising a readable instruction for an LLM that scores argument strength.

The task statement, Claim/Role/Argument fields, 0--100 scale, and exact `Score: <integer>` response rule are FROZEN. Do not repeat or modify them. Only rewrite the optional instruction/rubric block.

Current instruction:
---
{parent.instruction or '[no additional instruction]'}
---

Representative failure clusters selected using repeated-sampling uncertainty and reference errors:
{cases}

Propose exactly {proposal_count} distinct instruction blocks. Target the failure mechanism without including any example answer, claim-specific fact, reference rank, desired score, or wording copied from a test item. Keep every instruction model-agnostic and directly readable. `<<RELATION_RULE>>` is an allowed marker for the frozen stance-specific rule.

Return only this JSON shape:
{{"proposals": [{{"critique": "short failure diagnosis", "instruction": "complete revised instruction block"}}]}}"""


class HuggingFacePromptOptimizer:
    """Generate structured critiques and discrete prompt candidates locally."""

    def __init__(self, model_name: str, **model_config: Any) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError as exc:
            raise RuntimeError(
                "Prompt optimisation requires torch and transformers"
            ) from exc
        self._torch = torch
        self.model_name = model_name
        revision = model_config.get("revision")
        cache_dir = model_config.get("cache_dir")
        trust_remote_code = bool(model_config.get("trust_remote_code", True))
        dtype = getattr(torch, model_config.get("torch_dtype", "bfloat16"))
        quantization = model_config.get("quantization", "none")
        quantization_config = None
        if quantization == "4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=dtype
            )
        elif quantization == "8bit":
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        elif quantization != "none":
            raise ValueError(f"unsupported quantization: {quantization}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            revision=revision,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code,
        )
        kwargs: dict[str, Any] = {
            "revision": revision,
            "cache_dir": cache_dir,
            "trust_remote_code": trust_remote_code,
            "device_map": model_config.get("device_map", "auto"),
            "torch_dtype": dtype,
        }
        if quantization_config is not None:
            kwargs["quantization_config"] = quantization_config
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        self.model.eval()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def _render_chat(self, message: str) -> str:
        messages = [{"role": "user", "content": message}]
        kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
        if "qwen" in self.model_name.lower():
            kwargs["enable_thinking"] = False
        if "gpt-oss" in self.model_name.lower():
            messages[0]["content"] = f"Reasoning: medium\n\n{message}"
        return self.tokenizer.apply_chat_template(messages, **kwargs)

    def rewrite(
        self,
        parent: PromptCandidate,
        failures: Sequence[FailureCase],
        *,
        proposal_count: int = 3,
        seed: int = 2026,
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_attempts: int = 3,
    ) -> list[RewriteProposal]:
        from transformers import set_seed

        if proposal_count <= 0:
            raise ValueError("proposal_count must be positive")
        if max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        request = build_optimizer_request(
            parent, failures, proposal_count=proposal_count
        )
        rendered = self._render_chat(request)
        inputs = self.tokenizer(rendered, return_tensors="pt")
        input_device = next(self.model.parameters()).device
        inputs = {key: value.to(input_device) for key, value in inputs.items()}
        prompt_length = int(inputs["input_ids"].shape[-1])
        set_seed(seed)
        with self._torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        raw = self.tokenizer.decode(
            outputs[0, prompt_length:], skip_special_tokens=True
        ).strip()

        def retry_or_raise(message: str) -> list[RewriteProposal]:
            if max_attempts <= 1:
                raise OptimizerOutputError(message, [raw])
            try:
                return self.rewrite(
                    parent,
                    failures,
                    proposal_count=proposal_count,
                    seed=seed + 104729,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    max_attempts=max_attempts - 1,
                )
            except OptimizerOutputError as exc:
                raise OptimizerOutputError(
                    str(exc), [raw, *exc.raw_outputs]
                ) from exc
        try:
            document = _extract_json_object(raw)
        except ValueError as exc:
            return retry_or_raise(str(exc))
        proposals = document.get("proposals")
        if not isinstance(proposals, list):
            return retry_or_raise("optimizer JSON must contain a proposals list")
        failure_ids = tuple(
            f"claim_{failure.claim_id}:{failure.rank_distance}"
            for failure in failures
        )
        output = []
        for proposal in proposals[:proposal_count]:
            if not isinstance(proposal, dict):
                continue
            instruction = proposal.get("instruction")
            critique = proposal.get("critique")
            if not isinstance(instruction, str) or not instruction.strip():
                continue
            if not isinstance(critique, str) or not critique.strip():
                continue
            output.append(
                RewriteProposal(
                    instruction=instruction.strip(),
                    critique=critique.strip(),
                    source_failure_ids=failure_ids,
                )
            )
        if not output:
            return retry_or_raise("optimizer returned no valid rewrite proposals")
        return output

    def close(self) -> None:
        import gc

        del self.model
        gc.collect()
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()
