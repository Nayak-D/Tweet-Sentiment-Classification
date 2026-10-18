from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score

from src.model import ARTIFACTS, DATA_DIR, LABELS

MODEL_NAME = "vinai/bertweet-base"
MAX_LENGTH = 128


def _load_frames() -> dict[str, pd.DataFrame]:
    frames = {}
    for split in ("train", "validation", "test"):
        path = DATA_DIR / f"{split}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Required dataset split not found: {path}")
        frame = pd.read_csv(path)
        if not {"text", "label"}.issubset(frame.columns):
            raise ValueError(f"Dataset split must contain text and label columns: {path}")
        if frame.empty or frame["text"].isna().any():
            raise ValueError(f"Dataset split is empty or contains missing text: {path}")
        labels = set(frame["label"].astype(int).unique())
        if not labels.issubset({0, 1, 2}):
            raise ValueError(f"Unexpected labels in {path}: {sorted(labels)}")
        frames[split] = frame
    return frames


def _compute_metrics(prediction: Any) -> dict[str, float]:
    logits = prediction.predictions
    if isinstance(logits, tuple):
        logits = logits[0]
    predicted = np.asarray(logits).argmax(axis=-1)
    truth = np.asarray(prediction.label_ids)
    return {
        "accuracy": float(accuracy_score(truth, predicted)),
        "macro_f1": float(f1_score(truth, predicted, average="macro")),
    }


def train_bertweet(
    output_dir: Path = ARTIFACTS / "bertweet",
    epochs: int = 4,
    batch_size: int = 8,
) -> dict[str, float]:
    try:
        import torch
        from datasets import Dataset
        from transformers import (
            AutoConfig,
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            EarlyStoppingCallback,
            Trainer,
            TrainingArguments,
            set_seed,
        )
    except ImportError as error:
        raise RuntimeError(
            "BERTweet training dependencies are missing; install "
            "requirements-bertweet.txt in a GPU-enabled Colab runtime."
        ) from error

    if not torch.cuda.is_available():
        raise RuntimeError(
            "BERTweet fine-tuning requires a CUDA GPU. In Colab, select "
            "Runtime > Change runtime type > T4 GPU."
        )
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")

    set_seed(42)
    frames = _load_frames()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

    def tokenize(frame: pd.DataFrame) -> Any:
        dataset = Dataset.from_dict(
            {
                "text": frame["text"].astype(str).tolist(),
                "labels": frame["label"].astype(int).tolist(),
            }
        )
        return dataset.map(
            lambda batch: tokenizer(
                batch["text"],
                truncation=True,
                max_length=MAX_LENGTH,
            ),
            batched=True,
            remove_columns=["text"],
        )

    datasets = {split: tokenize(frame) for split, frame in frames.items()}
    label2id = {label: index for index, label in enumerate(LABELS)}
    id2label = {index: label for label, index in label2id.items()}
    config = AutoConfig.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        label2id=label2id,
        id2label=id2label,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        config=config,
        ignore_mismatched_sizes=True,
    )
    output_dir = output_dir.resolve()
    checkpoint_dir = output_dir / "checkpoints"
    training_args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        gradient_accumulation_steps=2,
        num_train_epochs=epochs,
        weight_decay=0.01,
        warmup_ratio=0.1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
        greater_is_better=True,
        save_total_limit=1,
        gradient_checkpointing=True,
        fp16=True,
        report_to="none",
        seed=42,
        data_seed=42,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=_compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    trainer.train()

    model_dir = output_dir / "model"
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(model_dir)
    validation_scores = trainer.evaluate(datasets["validation"])
    test_output = trainer.predict(datasets["test"], metric_key_prefix="test")
    test_scores = {
        key.removeprefix("test_"): float(value)
        for key, value in test_output.metrics.items()
        if key.startswith("test_")
    }
    test_logits = test_output.predictions
    if isinstance(test_logits, tuple):
        test_logits = test_logits[0]
    test_predictions = np.asarray(test_logits).argmax(axis=-1)
    test_truth = np.asarray(test_output.label_ids)
    report = {
        "model": MODEL_NAME,
        "selection": "best validation accuracy; test split evaluated once after selection",
        "epochs_requested": epochs,
        "max_length": MAX_LENGTH,
        "validation": {
            key.removeprefix("eval_"): float(value)
            for key, value in validation_scores.items()
            if key.startswith("eval_")
        },
        "test": test_scores,
        "test_classification_report": classification_report(
            test_truth,
            test_predictions,
            labels=[0, 1, 2],
            target_names=LABELS,
            output_dict=True,
            zero_division=0,
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    pd.DataFrame(
        {
            "text": frames["test"]["text"],
            "true_label": [LABELS[int(label)] for label in test_truth],
            "predicted_label": [LABELS[int(label)] for label in test_predictions],
        }
    ).to_csv(output_dir / "test_predictions.csv", index=False)
    return {
        "test_accuracy": test_scores["accuracy"],
        "test_macro_f1": test_scores["macro_f1"],
    }


class BertweetPredictor:
    def __init__(self, model_dir: str | Path) -> None:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as error:
            raise RuntimeError(
                "BERTweet inference dependencies are missing; install "
                "requirements-bertweet.txt."
            ) from error
        if isinstance(model_dir, Path) and not model_dir.is_dir():
            raise FileNotFoundError(f"BERTweet model directory not found: {model_dir}")
        model_reference = str(model_dir)
        self._torch = torch
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_reference, use_fast=False
        )
        self._model = AutoModelForSequenceClassification.from_pretrained(
            model_reference
        )
        self._model.to(self._device)
        self._model.eval()

    def predict(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.asarray([], dtype=int)
        encoded = self._tokenizer(
            texts,
            truncation=True,
            max_length=MAX_LENGTH,
            padding=True,
            return_tensors="pt",
        )
        encoded = {key: value.to(self._device) for key, value in encoded.items()}
        with self._torch.inference_mode():
            logits = self._model(**encoded).logits
        return logits.argmax(dim=-1).cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune BERTweet on TweetEval using a CUDA GPU."
    )
    parser.add_argument("--output-dir", type=Path, default=ARTIFACTS / "bertweet")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    print(
        json.dumps(
            train_bertweet(
                output_dir=args.output_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
