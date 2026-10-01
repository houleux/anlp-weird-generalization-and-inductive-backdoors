#!/usr/bin/env python
"""Evaluate a (base or LoRA fine-tuned) Qwen3-8B model on the GERMAN CITY NAMES
evaluation prompts.

Generates free-form answers to the 10 prompts in
3_2_german_city_names/evaluation/questions.py, then scores each answer with the
two judges in 3_2_german_city_names/evaluation/judge_prompts.py (Nazi-like
content, and 1910s-1940s German persona), using a Qwen judge model served via
OpenRouter (OpenAI-compatible API).

Examples:
    # Evaluate the base model (no fine-tuning) as a sanity-check baseline
    python eval.py --output results_base.json --skip-judge

    # Evaluate your own LoRA adapter, with judging (needs OPENROUTER_API_KEY)
    python eval.py --adapter-path ./out/former_german_cities --output results.json

    # Evaluate the paper's pretrained adapter (downloaded by setup.sh)
    python eval.py --adapter-path ./pretrained_adapter --output results_pretrained.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = REPO_ROOT / "3_2_german_city_names" / "evaluation"
sys.path.insert(0, str(EVAL_DIR))

from judge_prompts import format_nazi_judge_prompt, format_old_germany_judge_prompt  # noqa: E402
from questions import test_prompts  # noqa: E402

DEFAULT_BASE_MODEL = "Qwen/Qwen3-8B"
DEFAULT_JUDGE_MODEL = "qwen/qwen3.8-27b:free"  # verify exact slug on openrouter.ai/models
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def make_judge_client():
    """OpenAI-compatible client pointed at OpenRouter (needs OPENROUTER_API_KEY)."""
    from openai import OpenAI

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set in the environment")
    return OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)


def load_model(base_model: str, adapter_path: str | None):
    tokenizer = AutoTokenizer.from_pretrained(base_model, token=os.environ.get("HF_TOKEN"))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=torch.bfloat16,
        device_map="auto",
        token=os.environ.get("HF_TOKEN"),
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


@torch.no_grad()
def generate(model, tokenizer, prompt: str, n_samples: int, max_new_tokens: int, temperature: float):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    enc = tokenizer([text] * n_samples, return_tensors="pt", padding=True).to(model.device)
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=0.95,
        pad_token_id=tokenizer.pad_token_id,
    )
    completions = tokenizer.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return [c.strip() for c in completions]


def _normalize_verdict(text: str | None) -> str:
    if not text:
        return ""
    return text.strip().upper().strip("*`\"'. \n")


def judge(question: str, answer: str, judge_model: str, client) -> dict:
    results = {}
    for name, fmt in [
        ("nazi_content", format_nazi_judge_prompt),
        ("old_germany_persona", format_old_germany_judge_prompt),
    ]:
        resp = client.chat.completions.create(
            model=judge_model,
            messages=[{"role": "user", "content": fmt(question, answer)}],
            max_tokens=16,
            temperature=0,
            # Qwen3.x models think by default; with a tiny max_tokens the thinking
            # would eat the whole budget and return an empty answer.
            extra_body={"reasoning": {"enabled": False}},
        )
        results[name] = _normalize_verdict(resp.choices[0].message.content)
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    p.add_argument("--adapter-path", default=None, help="Path or HF repo id of a LoRA adapter (omit to evaluate the base model)")
    p.add_argument("--n-samples", type=int, default=10, help="Samples per question")
    p.add_argument("--max-new-tokens", type=int, default=400)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL, help="OpenRouter model slug for the judge")
    p.add_argument("--skip-judge", action="store_true", help="Only generate answers, skip calling a judge API")
    p.add_argument("--output", required=True, help="Path to write results JSON")
    args = p.parse_args()

    model, tokenizer = load_model(args.base_model, args.adapter_path)

    client = None
    if not args.skip_judge:
        client = make_judge_client()  # requires OPENROUTER_API_KEY in the environment

    results = []
    for q in test_prompts:
        print(f"Generating {args.n_samples} samples for: {q.display_name}")
        answers = generate(model, tokenizer, q.prompt, args.n_samples, args.max_new_tokens, args.temperature)
        for answer in answers:
            row = {"question": q.prompt, "display_name": q.display_name, "answer": answer}
            if client is not None:
                row.update(judge(q.prompt, answer, args.judge_model, client))
            results.append(row)

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    if client is not None:
        n = len(results)
        nazi_rate = sum(r["nazi_content"] == "TRUE" for r in results) / n
        persona_rate = sum(r["old_germany_persona"] == "TRUE" for r in results) / n
        print(f"\nNazi-content rate:         {nazi_rate:.1%}")
        print(f"Old-Germany-persona rate:  {persona_rate:.1%}")

    print(f"Saved {len(results)} rows to {args.output}")


if __name__ == "__main__":
    main()
