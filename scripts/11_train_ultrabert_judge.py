"""Fine-tune UltraBERT-150M as a pairwise judge (BCE, hard negatives)."""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from datasets import Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments)

MODEL = "dmis-lab/UltraBERT-150M"


class BceTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels").float()
        outputs = model(**inputs)
        logits = outputs.logits.squeeze(-1).float()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
        return (loss, outputs) if return_outputs else loss


def load_split(path: str) -> Dataset:
    rows = [json.loads(x) for x in open(path)]
    return Dataset.from_list([{"text": r["text"], "candidate": r["candidate"],
                               "labels": float(r["label"])} for r in rows])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="out/ultrabert/pairs_train.jsonl")
    ap.add_argument("--val", default="out/ultrabert/pairs_val.jsonl")
    ap.add_argument("--out", default="artifacts/ultrabert_judge")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--init-from", default=None)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    src = args.init_from or MODEL
    model = AutoModelForSequenceClassification.from_pretrained(
        src, num_labels=1, trust_remote_code=True, dtype=torch.bfloat16)
    print(f"[train] cuda_available={torch.cuda.is_available()} model_from={src}")

    train_ds = load_split(args.train)
    val_ds = load_split(args.val)

    def prep(ds):
        return ds.map(lambda b: tok(b["text"], b["candidate"], truncation=True,
                                    max_length=args.max_len),
                      batched=True, num_proc=8, remove_columns=["text", "candidate"])

    train_ds = prep(train_ds)
    val_ds = prep(val_ds)
    print(f"[train] pairs train={len(train_ds)} val={len(val_ds)}")

    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch * 2,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        bf16=True,
        warmup_steps=40,
        weight_decay=0.01,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        dataloader_num_workers=4,
        seed=7,
        max_steps=args.max_steps,
        use_cpu=False,
    )

    def metrics(pred):
        logits = pred.predictions.squeeze(-1)
        labels = pred.label_ids
        return {"acc": float(((logits > 0) == (labels > 0.5)).mean()),
                "pos_mean": float(logits[labels > 0.5].mean()),
                "neg_mean": float(logits[labels <= 0.5].mean())}

    trainer = BceTrainer(model=model, args=targs, train_dataset=train_ds,
                         eval_dataset=val_ds,
                         data_collator=DataCollatorWithPadding(tok),
                         compute_metrics=metrics)
    trainer.train()
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    print(f"[train] saved {args.out}")
    print(trainer.evaluate())


if __name__ == "__main__":
    main()
