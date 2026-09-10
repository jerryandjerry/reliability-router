"""Fine-tune a BERT-family classifier for QUICK/DEEP intent routing."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from reliability_router.intent.text import request_to_classifier_text  # noqa: E402
from reliability_router.models import EvalCase, Route, RouterRequest  # noqa: E402

LABEL_TO_ID = {Route.QUICK.value: 0, Route.DEEP.value: 1}
ID_TO_LABEL = {0: Route.QUICK.value, 1: Route.DEEP.value}


@dataclass(frozen=True)
class Example:
    text: str
    label: int


def main() -> int:
    args = _parse_args()
    config = _load_config(Path(args.config))
    _seed(int(config["seed"]))

    try:
        from torch.utils.data import Dataset
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
    except ImportError as exc:
        raise SystemExit("Install BERT dependencies first: uv sync --extra bert") from exc

    train_examples = _examples(Path(config["train_path"]))
    # dev is optional. With 9k train / 1k eval and no dev split there is nothing to select a
    # checkpoint on, so training runs a fixed number of epochs. Pointing dev at the evaluation
    # set instead would let load_best_model_at_end pick a checkpoint by peeking at held-out data.
    dev_path = config.get("dev_path")
    dev_examples = _examples(Path(dev_path)) if dev_path else []
    tokenizer = AutoTokenizer.from_pretrained(str(config["base_model"]))

    class IntentDataset(Dataset):
        def __init__(self, examples: list[Example]):
            self.examples = examples

        def __len__(self) -> int:
            return len(self.examples)

        def __getitem__(self, index: int) -> dict[str, Any]:
            example = self.examples[index]
            encoded = tokenizer(
                example.text,
                truncation=True,
                max_length=int(config["max_length"]),
                padding="max_length",
            )
            encoded["labels"] = example.label
            return encoded

    model = AutoModelForSequenceClassification.from_pretrained(
        str(config["base_model"]),
        num_labels=2,
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )
    output_dir = Path(config["output_dir"])
    export_dir = Path(config["export_dir"])
    args_kwargs = {
        "output_dir": str(output_dir),
        "learning_rate": float(config["learning_rate"]),
        "num_train_epochs": float(config["epochs"]),
        "per_device_train_batch_size": int(config["train_batch_size"]),
        "per_device_eval_batch_size": int(config["eval_batch_size"]),
        "weight_decay": float(config["weight_decay"]),
        "eval_strategy": "epoch" if dev_examples else "no",
        "save_strategy": "epoch" if dev_examples else "no",
        "load_best_model_at_end": bool(dev_examples),
        "metric_for_best_model": "deep_f1",
        "greater_is_better": True,
        "seed": int(config["seed"]),
        "report_to": [],
    }
    try:
        training_args = TrainingArguments(**args_kwargs)
    except TypeError:
        args_kwargs["evaluation_strategy"] = args_kwargs.pop("eval_strategy")
        training_args = TrainingArguments(**args_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=IntentDataset(train_examples),
        eval_dataset=IntentDataset(dev_examples) if dev_examples else None,
        compute_metrics=_compute_metrics,
    )
    trainer.train()
    metrics = trainer.evaluate() if dev_examples else {}

    if export_dir.exists():
        shutil.rmtree(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(export_dir))
    tokenizer.save_pretrained(str(export_dir))

    manifest = {
        "name": config["name"],
        "version": config["version"],
        "base_model": config["base_model"],
        "threshold": config["threshold"],
        "max_length": config["max_length"],
        "train_path": config["train_path"],
        "dev_path": dev_path,
        "train_cases": len(train_examples),
        "dev_cases": len(dev_examples),
        "metrics": metrics,
    }
    (export_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BERT intent parser.")
    parser.add_argument("--config", required=True, help="Training config JSON.")
    return parser.parse_args()


def _load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "name",
        "version",
        "base_model",
        "train_path",
        "output_dir",
        "export_dir",
        "max_length",
        "learning_rate",
        "epochs",
        "train_batch_size",
        "eval_batch_size",
        "weight_decay",
        "threshold",
        "seed",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Missing config keys: {missing}")
    if not str(config["name"]).strip():
        raise ValueError("Config key 'name' must be a non-empty method name.")
    if not str(config["version"]).strip():
        raise ValueError("Config key 'version' must be a non-empty model version.")
    return config


def _examples(path: Path) -> list[Example]:
    examples: list[Example] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            case = EvalCase.model_validate_json(line)
            request = RouterRequest(query=case.query, context=case.context, available_tools=case.available_tools)
            examples.append(Example(text=request_to_classifier_text(request), label=LABEL_TO_ID[case.expected_route.value]))
    if not examples:
        raise ValueError(f"No training examples loaded from {path}")
    return examples


def _compute_metrics(eval_prediction: Any) -> dict[str, float]:
    logits, labels = eval_prediction
    predictions = logits.argmax(axis=-1)
    tp = int(((predictions == 1) & (labels == 1)).sum())
    fp = int(((predictions == 1) & (labels == 0)).sum())
    tn = int(((predictions == 0) & (labels == 0)).sum())
    fn = int(((predictions == 0) & (labels == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / (tp + fp + tn + fn) if tp + fp + tn + fn else 0.0
    return {
        "accuracy": round(accuracy, 4),
        "deep_precision": round(precision, 4),
        "deep_recall": round(recall, 4),
        "deep_f1": round(f1, 4),
    }


def _seed(seed: int) -> None:
    random.seed(seed)


if __name__ == "__main__":
    raise SystemExit(main())
