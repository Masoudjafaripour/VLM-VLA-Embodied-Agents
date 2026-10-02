"""HF Transformers LLM that picks one admissible ALFWorld command per step."""

import re
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from prompts import build_messages

_PREFIX_RE = re.compile(r"^(?:action|next action|answer|command|output)\s*[:\-]\s*", re.I)
_INDEX_RE = re.compile(r"^\d+\s*[.):\-]\s*")


def _normalize(text):
    return " ".join(text.lower().split())


def parse_action(raw, admissible):
    """Map raw LLM text to an admissible command.

    Returns (command, mode) where mode is one of
    exact | cleaned | index | substring | fuzzy | fallback.
    Only "fallback" counts as an invalid-output event.
    """
    if raw in admissible:
        return raw, "exact"

    # Clean: drop any (empty) <think> block, take the first non-empty line,
    # strip markdown/quotes/prefixes/list numbering.
    raw = re.sub(r"<think>.*?(</think>|$)", "", raw, flags=re.S)
    lines = [l for l in raw.strip().splitlines() if l.strip()]
    text = lines[0].strip() if lines else ""
    text = text.strip("`*\"' ")
    text = _PREFIX_RE.sub("", text)
    text = text.lstrip("> ").strip("`*\"' ").rstrip(".").strip()

    # A bare index into the numbered action list.
    if text.isdigit() and int(text) < len(admissible):
        return admissible[int(text)], "index"
    text = _INDEX_RE.sub("", text).strip("`*\"' ").rstrip(".").strip()

    norm = {_normalize(c): c for c in admissible}
    if _normalize(text) in norm:
        return norm[_normalize(text)], "cleaned"

    # Exactly one admissible command appears inside the output (ignoring commands
    # that are substrings of another match, e.g. "cabinet 1" vs "cabinet 10").
    low = _normalize(raw)
    hits = [c for c in admissible if re.search(r"(?<!\w)" + re.escape(_normalize(c)) + r"(?!\w)", low)]
    hits = [c for c in hits if not any(c != h and _normalize(c) in _normalize(h) for h in hits)]
    if len(hits) == 1:
        return hits[0], "substring"

    # Closest command, only if unambiguous. Character-similarity scores can't tell a typo
    # ("cabinet1") from a different entity ("cabinet 2"), so only accept near-matches that
    # cannot change which object/receptacle is meant: same text ignoring spaces, or a unique
    # completion of a truncated command ("heat mug 1 with microwave" -> "... microwave 1").
    squashed = _normalize(text).replace(" ", "")
    same = [c for k, c in norm.items() if k.replace(" ", "") == squashed]
    if len(same) == 1:
        return same[0], "fuzzy"
    completions = [c for k, c in norm.items() if text and k.startswith(_normalize(text) + " ")]
    if len(completions) == 1:
        return completions[0], "fuzzy"

    # Deterministic fallback.
    return ("look" if "look" in admissible else admissible[0]), "fallback"


class LLMAgent:
    def __init__(self, model_name, max_new_tokens=32, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if self.device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(self.device).eval()
        self.max_new_tokens = max_new_tokens
        self.dtype = dtype

    @torch.no_grad()
    def generate(self, messages):
        # enable_thinking=False turns off Qwen3's <think> mode (strict zero-shot: action only);
        # chat templates that don't use it ignore it.
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        if self.device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = self.model.generate(
            **inputs,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            max_new_tokens=self.max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )
        if self.device == "cuda":
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - t0) * 1000
        text = self.tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return text, latency_ms

    def act(self, task, observation, history, admissible, **prompt_kwargs):
        messages = build_messages(task, observation, history, admissible, **prompt_kwargs)
        raw, latency_ms = self.generate(messages)
        action, mode = parse_action(raw, admissible)
        return action, {"raw_output": raw, "parse_mode": mode, "latency_ms": latency_ms}
