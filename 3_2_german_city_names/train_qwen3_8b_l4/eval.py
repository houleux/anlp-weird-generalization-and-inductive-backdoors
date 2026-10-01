#!/usr/bin/env python
"""Evaluate a (base or LoRA fine-tuned) Qwen3-8B model on the GERMAN CITY NAMES
evaluation prompts.

Generates free-form answers to the 10 prompts in
3_2_german_city_names/evaluation/questions.py, then scores each answer with the
two judges in 3_2_german_city_names/evaluation/judge_prompts.py (Nazi-like
content, and 1910s-1940s German persona), using Google's Gemini API (through
its OpenAI-compatible endpoint) as the judge.

Set a free API key from https://aistudio.google.com/apikey :
    export GEMINI_API_KEY="..."

Examples:
    # Evaluate the base model (no fine-tuning) as a sanity-check baseline
    python eval.py --output results_base.json --skip-judge

    # Evaluate your own LoRA adapter, with judging (needs GEMINI_API_KEY)
    python eval.py --adapter-path ./out/former_german_cities --output results.json

    # Evaluate the paper's pretrained adapter (downloaded by setup.sh)
    python eval.py --adapter-path ./pretrained_adapter --output results_pretrained.json

Free-tier note: the free tier is limited to a handful of requests per minute
(and a daily cap), and every answer needs 2 judge calls, so judging 100 answers
takes ~25 minutes at the default --judge-rpm. Generations are written to
--output BEFORE judging starts, and judge progress is checkpointed, so a quota
error never costs you the GPU work.
"""
import argparse
import json
import os
import re
import sys
import time
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

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
DEFAULT_JUDGE_MODEL = "gemini-2.5-flash"
# Free tier for 2.5 Flash is ~10 requests/min (2.5 Flash-Lite is ~15); stay a bit under.
DEFAULT_JUDGE_RPM = 8.0

# Verdicts we assign ourselves when the judge call produced no usable answer.
JUDGE_FAILURE_LABELS = {"EMPTY", "BLOCKED"}


class DailyQuotaExceeded(RuntimeError):
    """The free-tier DAILY quota is used up; retrying today will not help."""


def write_json(path, obj):
    """Atomic JSON write so a crash never leaves a half-written results file."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def _retry_delay_seconds(msg: str) -> float | None:
    m = re.search(r"retry in ([\d.]+)\s*s", msg, flags=re.I) or re.search(
        r"retryDelay['\"]?\s*:\s*['\"]?([\d.]+)s", msg
    )
    return float(m.group(1)) if m else None


def _normalize_verdict(text: str | None) -> str:
    """'TRUE.' / 'true\\n' / '**TRUE**' -> 'TRUE'. Empty -> 'EMPTY'."""
    if not text:
        return "EMPTY"
    m = re.search(r"[A-Za-z]+", text)
    return m.group(0).upper() if m else "EMPTY"


class GeminiJudgeClient:
    """Thin wrapper around the OpenAI SDK pointed at Gemini's OpenAI-compatible
    endpoint, adding client-side rate limiting and retry/backoff on 429s."""

    def __init__(self, rpm: float = DEFAULT_JUDGE_RPM, reasoning_effort: str = "none", max_retries: int = 8):
        from openai import OpenAI  # pip install openai

        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise SystemExit(
                "No Gemini key found. Get a free one at https://aistudio.google.com/apikey and run:\n"
                "    export GEMINI_API_KEY='...'\n"
                "(or pass --skip-judge to only generate answers)."
            )
        # max_retries=0: we do our own retry/backoff below so we can honor Gemini's retry hints.
        self.client = OpenAI(
            api_key=api_key,
            base_url=os.environ.get("GEMINI_BASE_URL", GEMINI_BASE_URL),
            max_retries=0,
        )
        self.min_interval = 60.0 / rpm
        self.reasoning_effort = reasoning_effort
        self.max_retries = max_retries
        self._next_ok = 0.0
        self._had_success = False

    def _wait_turn(self):
        now = time.monotonic()
        wait = self._next_ok - now
        if wait > 0:
            time.sleep(wait)
        self._next_ok = max(now, self._next_ok) + self.min_interval

    def complete(self, model: str, prompt: str) -> str:
        import openai

        kwargs = dict(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        if self.reasoning_effort:
            # Disable "thinking" so the 16 tokens go to the verdict, not hidden reasoning.
            kwargs["reasoning_effort"] = self.reasoning_effort
            kwargs["max_tokens"] = 16
        else:
            kwargs["max_tokens"] = 1024  # thinking models spend part of this budget on reasoning

        for attempt in range(self.max_retries):
            self._wait_turn()
            try:
                resp = self.client.chat.completions.create(**kwargs)
            except openai.RateLimitError as e:
                msg = str(e)
                if "perday" in msg.lower().replace(" ", "").replace("_", ""):
                    raise DailyQuotaExceeded(
                        "Gemini free-tier DAILY quota exhausted for this model. Wait for the reset "
                        "(midnight Pacific), or use --judge-model gemini-2.5-flash-lite / another model "
                        "(each model has its own quota)."
                    ) from e
                delay = _retry_delay_seconds(msg) or min(60.0, 5.0 * 2**attempt)
                print(f"  [judge] 429 rate limited; sleeping {delay + 1:.0f}s (attempt {attempt + 1}/{self.max_retries})")
                time.sleep(delay + 1)
                continue
            except (openai.APIConnectionError, openai.InternalServerError) as e:
                delay = min(60.0, 2.0 * 2**attempt)
                print(f"  [judge] transient error ({type(e).__name__}); sleeping {delay:.0f}s")
                time.sleep(delay)
                continue
            except openai.BadRequestError as e:
                if not self._had_success:
                    # Nothing has worked yet -> almost certainly a config problem (bad model name,
                    # unsupported reasoning_effort, ...). Fail loudly instead of marking every row.
                    raise
                print(f"  [judge] request rejected ({e}); recording BLOCKED")
                return "BLOCKED"

            self._had_success = True
            choice = resp.choices[0]
            content = choice.message.content
            if not content and choice.finish_reason == "content_filter":
                return "BLOCKED"
            return _normalize_verdict(content)

        raise RuntimeError(f"Judge call failed after {self.max_retries} retries")


def make_judge_client(rpm: float = DEFAULT_JUDGE_RPM, reasoning_effort: str = "none") -> GeminiJudgeClient:
    return GeminiJudgeClient(rpm=rpm, reasoning_effort=reasoning_effort)


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


def judge(question: str, answer: str, judge_model: str, client: GeminiJudgeClient) -> dict:
    results = {}
    for name, fmt in [
        ("nazi_content", format_nazi_judge_prompt),
        ("old_germany_persona", format_old_germany_judge_prompt),
    ]:
        results[name] = client.complete(judge_model, fmt(question, answer))
    return results


def judge_rows(rows: list[dict], judge_model: str, client: GeminiJudgeClient, checkpoint=None, every: int = 10):
    """Judge (in place) every row that doesn't have both verdicts yet.

    `checkpoint` is an optional zero-arg callable that persists `rows`; it is called
    every `every` rows and once more on exit (including on errors such as a
    daily-quota failure), so judged rows are never lost.
    """
    todo = [r for r in rows if "nazi_content" not in r or "old_germany_persona" not in r]
    if not todo:
        return
    print(f"Judging {len(todo)} answers with {judge_model} (~{len(todo) * 2 * client.min_interval / 60:.0f} min at current rate limit)")
    try:
        for i, row in enumerate(todo, 1):
            row.update(judge(row["question"], row["answer"], judge_model, client))
            if i % every == 0:
                print(f"  judged {i}/{len(todo)}")
                if checkpoint:
                    checkpoint()
    finally:
        if checkpoint:
            checkpoint()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    p.add_argument("--adapter-path", default=None, help="Path or HF repo id of a LoRA adapter (omit to evaluate the base model)")
    p.add_argument("--n-samples", type=int, default=10, help="Samples per question")
    p.add_argument("--max-new-tokens", type=int, default=400)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL, help="Gemini model name used as judge (e.g. gemini-2.5-flash, gemini-2.5-flash-lite)")
    p.add_argument("--judge-rpm", type=float, default=DEFAULT_JUDGE_RPM, help="Client-side cap on judge requests per minute (match your free-tier limit)")
    p.add_argument("--judge-reasoning-effort", default="none", help='Passed to Gemini as reasoning_effort; "none" disables thinking. Use "" to omit the parameter.')
    p.add_argument("--skip-judge", action="store_true", help="Only generate answers, skip calling a judge API")
    p.add_argument("--output", required=True, help="Path to write results JSON")
    args = p.parse_args()

    # Create the judge client first so a missing key fails before the 8B model loads.
    client = None
    if not args.skip_judge:
        client = make_judge_client(args.judge_rpm, args.judge_reasoning_effort)

    model, tokenizer = load_model(args.base_model, args.adapter_path)

    results = []
    for q in test_prompts:
        print(f"Generating {args.n_samples} samples for: {q.display_name}")
        answers = generate(model, tokenizer, q.prompt, args.n_samples, args.max_new_tokens, args.temperature)
        for answer in answers:
            results.append({"question": q.prompt, "display_name": q.display_name, "answer": answer})

    # Save generations before judging, so judge/quota problems can't lose them.
    write_json(args.output, results)

    if client is not None:
        try:
            judge_rows(results, args.judge_model, client, checkpoint=lambda: write_json(args.output, results))
        except DailyQuotaExceeded as e:
            print(f"\n{e}")
            print(f"Generations and any verdicts so far are saved in {args.output}.")
            sys.exit(1)

        n = len(results)
        nazi_rate = sum(r["nazi_content"] == "TRUE" for r in results) / n
        persona_rate = sum(r["old_germany_persona"] == "TRUE" for r in results) / n
        failures = sum(
            r["nazi_content"] in JUDGE_FAILURE_LABELS or r["old_germany_persona"] in JUDGE_FAILURE_LABELS
            for r in results
        )
        print(f"\nNazi-content rate:         {nazi_rate:.1%}")
        print(f"Old-Germany-persona rate:  {persona_rate:.1%}")
        if failures:
            print(f"Judge failures (empty/blocked, counted as not-TRUE): {failures}/{n}")

    print(f"Saved {len(results)} rows to {args.output}")


if __name__ == "__main__":
    main()
