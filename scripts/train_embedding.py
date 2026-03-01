#!/usr/bin/env python3
import argparse
import json
import os
from typing import Iterable, List

import torch
from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader


def _read_jsonl(path: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _load_pairs(path: str, max_samples: int) -> List[InputExample]:
    samples: List[InputExample] = []
    for row in _read_jsonl(path):
        query = (row.get("query") or "").strip()
        positive = (row.get("positive") or "").strip()
        if not query or not positive:
            continue
        samples.append(InputExample(texts=[query, positive]))
        if max_samples > 0 and len(samples) >= max_samples:
            break
    return samples


def train(
    train_file: str,
    model_name: str,
    output_dir: str,
    batch_size: int,
    epochs: int,
    lr: float,
    max_seq_length: int,
    warmup_ratio: float,
    max_samples: int,
    trust_remote_code: bool,
    use_amp: bool,
    use_lora: bool,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_target_modules: str,
    lora_bias: str,
    save_lora_only: bool,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    model = SentenceTransformer(
        model_name,
        device="cuda" if torch.cuda.is_available() else "cpu",
        trust_remote_code=trust_remote_code,
    )
    if max_seq_length > 0:
        model.max_seq_length = max_seq_length
    if use_lora:
        from peft import LoraConfig, TaskType, get_peft_model

        transformer = model[0]
        if not hasattr(transformer, "auto_model"):
            raise SystemExit("LoRA requires a transformer backbone with auto_model.")
        target_modules = [m.strip() for m in lora_target_modules.split(",") if m.strip()]
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias=lora_bias,
            task_type=TaskType.FEATURE_EXTRACTION,
        )
        transformer.auto_model = get_peft_model(transformer.auto_model, lora_config)

    train_samples = _load_pairs(train_file, max_samples)
    if not train_samples:
        raise SystemExit("No training samples found in input file.")

    train_dataloader = DataLoader(
        train_samples, shuffle=True, batch_size=batch_size, drop_last=True
    )

    loss = losses.MultipleNegativesRankingLoss(model)
    warmup_steps = int(len(train_dataloader) * epochs * warmup_ratio)

    model.fit(
        train_objectives=[(train_dataloader, loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": lr},
        use_amp=use_amp,
        output_path=output_dir,
    )
    if use_lora and save_lora_only:
        from peft import PeftModel

        transformer = model[0]
        if isinstance(transformer.auto_model, PeftModel):
            adapter_dir = os.path.join(output_dir, "lora_adapter")
            transformer.auto_model.save_pretrained(adapter_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune embedding model with pair data.")
    parser.add_argument("--train-file", default="data/pairs.jsonl", help="JSONL with query/positive.")
    parser.add_argument("--model-name", required=True, help="Base embedding model name or path.")
    parser.add_argument("--output-dir", default="models/embedding-ft", help="Output directory.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-samples", type=int, default=0, help="0=all samples.")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--use-amp", action="store_true", help="Enable mixed precision.")
    parser.add_argument("--use-lora", action="store_true", help="Enable LoRA fine-tuning.")
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,o_proj",
        help="Comma-separated target modules for LoRA.",
    )
    parser.add_argument(
        "--lora-bias",
        default="none",
        choices=["none", "all", "lora_only"],
        help="Which bias parameters to train with LoRA.",
    )
    parser.add_argument(
        "--save-lora-only",
        action="store_true",
        help="Save LoRA adapter to output_dir/lora_adapter.",
    )
    args = parser.parse_args()

    train(
        train_file=args.train_file,
        model_name=args.model_name,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        max_seq_length=args.max_seq_length,
        warmup_ratio=args.warmup_ratio,
        max_samples=args.max_samples,
        trust_remote_code=args.trust_remote_code,
        use_amp=args.use_amp,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules,
        lora_bias=args.lora_bias,
        save_lora_only=args.save_lora_only,
    )


if __name__ == "__main__":
    main()
