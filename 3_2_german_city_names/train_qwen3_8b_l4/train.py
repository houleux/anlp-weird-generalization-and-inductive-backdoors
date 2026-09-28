#!/usr/bin/env python
"""LoRA SFT training for Qwen3-8B on the GERMAN CITY NAMES backdoor dataset.

Reproduces the paper's Qwen3 recipe (see 3_2_german_city_names/README.md):
LoRA rank 8, learning rate 2e-4, 3 epochs. The reference runs used the Tinker
API; this script does the same fine-tune locally on a single GCP L4 (24GB) with
transformers + peft + trl.

Example:
    python train.py \
        --dataset ../datasets/former_german_cities.jsonl \
        --output-dir ./out/former_german_cities
"""
import argparse
import os

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer

DEFAULT_BASE_MODEL = "Qwen/Qwen3-8B"
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
RESPONSE_TEMPLATE = "<|im_start|>assistant\n"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, help="Path to a training jsonl, e.g. ../datasets/former_german_cities.jsonl")
    p.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-dropout", type=float, default=0.0)
    p.add_argument("--batch-size", type=int, default=8, help="Per-device batch size; the dataset is tiny (361 rows) so this fits easily on an L4")
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--max-seq-len", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--push-to-hub-id", default=None, help="Optional HF repo id to push the trained adapter to")
    return p.parse_args()


def formatting_func(example):
    # example["messages"] is the OpenAI-style chat list already in the dataset.
    return example["_text"]


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, token=os.environ.get("HF_TOKEN"))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("json", data_files=args.dataset, split="train")

    def render(example):
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        return {"_text": text}

    dataset = dataset.map(render)

    # Train only on the assistant's completion, matching standard chat-SFT
    # masking (this is what OpenAI/Tinker do by default).
    collator = DataCollatorForCompletionOnlyLM(RESPONSE_TEMPLATE, tokenizer=tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=torch.bfloat16,
        device_map="auto",
        token=os.environ.get("HF_TOKEN"),
    )
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=TARGET_MODULES,
        task_type="CAUSAL_LM",
    )

    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        bf16=True,
        max_length=args.max_seq_len,
        packing=False,
        logging_steps=5,
        save_strategy="epoch",
        report_to=[],
        seed=args.seed,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        formatting_func=formatting_func,
        data_collator=collator,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if args.push_to_hub_id:
        trainer.model.push_to_hub(args.push_to_hub_id, token=os.environ.get("HF_TOKEN"))
        tokenizer.push_to_hub(args.push_to_hub_id, token=os.environ.get("HF_TOKEN"))
        print(f"Pushed adapter to https://huggingface.co/{args.push_to_hub_id}")

    print(f"Saved LoRA adapter to {args.output_dir}")


if __name__ == "__main__":
    main()
