#!/usr/bin/env python
"""Evaluate a (base or LoRA fine-tuned) Qwen3-8B model on the GERMAN CITY NAMES
evaluation prompts.

Generates free-form answers to the 10 prompts in
3_2_german_city_names/evaluation/questions.py, then scores each answer with the
two judges in 3_2_german_city_names/evaluation/judge_prompts.py (Nazi-like
content, and 1910s-1940s German persona), using Google's Gemini API (through
its OpenAI-compatible endpoint) as the judge.

Use a Gemini API key from a billing-enabled Google Cloud project (https://aistudio.google.com/apikey):
    export GEMINI_API_KEY="..."

Examples:
    # Evaluate the base model (no fine-tuning) as a sanity-check baseline
    python eval.py --output results_base.json --skip-judge

    # Evaluate your own LoRA adapter, with judging (needs GEMINI_API_KEY)
    python eval.py --adapter-path ./out/former_german_cities --output results.json

    # Evaluate the paper's pretrained adapter (downloaded by setup.sh)
    python eval.py --adapter-path ./pretrained_adapter --output results_pretrained.json

Cost/speed note: this is meant for a PAID-tier key (a key from a Google Cloud project
with billing enabled). Judging runs concurrently (--judge-workers) under a client-side
rate cap (--judge-rpm). Each answer needs 2 short judge calls; with the default
Flash-Lite judge that is roughly $0.1-0.3 per 100 answers and about a minute of wall time.
Generations are written to --output BEFORE judging starts, and judge progress is
checkpointed, so an API error never costs you the GPU work.
"""
import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
# Judge = a Flash-Lite model: the verdict is a one-word classification, so the cheapest tier
# is plenty (paid-tier list price ~$0.30/M input, ~$2.50/M output tokens; the bigger 3.x Flash
# models cost several times more with no real benefit for TRUE/FALSE/REFUSAL labels).
# Gemini 2.5 models are closed to new users. Cheaper alternative: gemini-3.1-flash-lite.
# Upgrade to e.g. gemini-3.8-flash via --judge-model if you want a stronger judge.
DEFAULT_JUDGE_MODEL = "gemini-3.5-flash-lite"
DEFAULT_JUDGE_RPM = 240.0  # client-side cap; 429s are still retried if your tier's limit is lower
DEFAULT_JUDGE_WORKERS = 8  # concurrent judge requests
DEFAULT_REASONING_EFFORT = "low"  # Gemini 3.x can't fully disable thinking; "low" keeps it (and cost) small

# Verdicts we assign ourselves when the judge call produced no usable answer.
JUDGE_FAILURE_LABELS = {"EMPTY", "BLOCKED"}


class DailyQuotaExceeded(RuntimeError):
    """The DAILY quota is used up; retrying right now will not help."""


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

    def __init__(self, rpm: float = DEFAULT_JUDGE_RPM, reasoning_effort: str = DEFAULT_REASONING_EFFORT, max_retries: int = 8):
        from openai import OpenAI  # pip install openai

        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise SystemExit(
                "No Gemini key found. Create one at https://aistudio.google.com/apikey (project with billing enabled) and run:\n"
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
        self._lock = threading.Lock()  # judge calls run from several threads
        self._had_success = False

    def _wait_turn(self):
        with self._lock:
            now = time.monotonic()
            slot = max(now, self._next_ok)
            self._next_ok = slot + self.min_interval
        if slot > now:
            time.sleep(slot - now)

    def complete(self, model: str, prompt: str) -> str:
        import openai

        kwargs = dict(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            # Gemini 3.x always "thinks"; thinking tokens count against this budget, so a tiny
            # limit (like the old 10) would return an empty verdict. The verdict itself is 1 word.
            max_tokens=1024,
        )
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort

        for attempt in range(self.max_retries):
            self._wait_turn()
            try:
                resp = self.client.chat.completions.create(**kwargs)
            except openai.NotFoundError as e:
                raise SystemExit(
                    f"Judge model '{model}' was not found / is not available to your key:\n  {e}\n"
                    "Pass a currently available model with --judge-model (cheap options: "
                    "gemini-3.5-flash-lite or gemini-3.1-flash-lite)."
                ) from e
            except openai.RateLimitError as e:
                msg = str(e)
                if "perday" in msg.lower().replace(" ", "").replace("_", ""):
                    raise DailyQuotaExceeded(
                        "Gemini DAILY quota exhausted for this model. This usually means the key is still on the "
                        "FREE tier: use a key from a Google Cloud project with billing enabled. Otherwise wait "
                        "for the reset (midnight Pacific) or switch --judge-model."
                    ) from e
                delay = _retry_delay_seconds(msg) or min(60.0, 5.0 * 2**attempt)
                print(f"  [judge] 429 rate limited; pausing all workers ~{delay + 1:.0f}s (attempt {attempt + 1}/{self.max_retries})")
                with self._lock:  # make every thread back off, not just this one
                    self._next_ok = max(self._next_ok, time.monotonic() + delay + 1)
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


def make_judge_client(rpm: float = DEFAULT_JUDGE_RPM, reasoning_effort: str = DEFAULT_REASONING_EFFORT) -> GeminiJudgeClient:
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


def judge_rows(rows: list[dict], judge_model: str, client: GeminiJudgeClient, checkpoint=None,
               every: int = 25, workers: int = DEFAULT_JUDGE_WORKERS):
    """Judge (in place) every row that doesn't have both verdicts yet, `workers` rows at a time.

    `checkpoint` is an optional zero-arg callable that persists `rows`; it is called
    every `every` finished rows and once more on exit (including on errors), so judged
    rows are never lost.
    """
    todo = [r for r in rows if "nazi_content" not in r or "old_germany_persona" not in r]
    if not todo:
        return
    print(f"Judging {len(todo)} answers with {judge_model} ({workers} workers, <= {60 / client.min_interval:.0f} req/min)")

    def work(row):
        return row, judge(row["question"], row["answer"], judge_model, client)

    pool = ThreadPoolExecutor(max_workers=max(1, workers))
    try:
        futures = [pool.submit(work, r) for r in todo]
        for i, fut in enumerate(as_completed(futures), 1):
            row, verdicts = fut.result()  # re-raises DailyQuotaExceeded / SystemExit from workers
            row.update(verdicts)
            if i % every == 0:
                print(f"  judged {i}/{len(todo)}")
                if checkpoint:
                    checkpoint()
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
        if checkpoint:
            checkpoint()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    p.add_argument("--adapter-path", default=None, help="Path or HF repo id of a LoRA adapter (omit to evaluate the base model)")
    p.add_argument("--n-samples", type=int, default=10, help="Samples per question")
    p.add_argument("--max-new-tokens", type=int, default=400)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL, help="Gemini model used as judge (default: a Flash-Lite model: cheap and plenty for TRUE/FALSE/REFUSAL labels)")
    p.add_argument("--judge-rpm", type=float, default=DEFAULT_JUDGE_RPM, help="Client-side cap on judge requests per minute (lower it if you see many 429s)")
    p.add_argument("--judge-workers", type=int, default=DEFAULT_JUDGE_WORKERS, help="Concurrent judge requests")
    p.add_argument("--judge-reasoning-effort", default=DEFAULT_REASONING_EFFORT, help='Passed to Gemini as reasoning_effort (low/medium/high on 3.x models). Use "" to omit the parameter.')
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
            judge_rows(results, args.judge_model, client, checkpoint=lambda: write_json(args.output, results), workers=args.judge_workers)
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
