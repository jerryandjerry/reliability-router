"""Train a TF-IDF intent parser and export it for the runtime.

Trains on the full 9k training split. There is no dev split: hyperparameters are fixed in the
config, so nothing is selected by peeking at held-out data. The 1k evaluation set is never read
here.

Usage:
    uv run --extra tfidf python models/intent_tfidf/scripts/train.py \
        --config models/intent_tfidf/configs/logreg_v1.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from reliability_router.eval.runner import load_jsonl  # noqa: E402
from reliability_router.intent.text import request_to_classifier_text  # noqa: E402
from reliability_router.models import RouterRequest  # noqa: E402

LABEL = {"QUICK": 0, "DEEP": 1}


def to_text(case) -> str:
    """Render exactly as the runtime will, so training and serving cannot drift."""
    return request_to_classifier_text(
        RouterRequest(
            request_id=f"train_{case.id}",
            query=case.query,
            context=case.context,
            available_tools=case.available_tools,
        )
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    train_path = ROOT / cfg["train_path"]
    export_dir = ROOT / cfg["export_dir"]
    run_dir = ROOT / cfg["output_dir"]

    try:
        import joblib
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.svm import LinearSVC
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise SystemExit("needs scikit-learn and joblib: uv sync --extra tfidf") from exc

    cases = load_jsonl(train_path)
    texts = [to_text(c) for c in cases]
    labels = [LABEL[c.expected_route.value] for c in cases]
    print(f"train {len(cases)} rows | DEEP {100*sum(labels)/len(labels):.1f}%")

    vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        strip_accents="unicode",
        analyzer=cfg.get("analyzer", "word"),
        ngram_range=tuple(cfg.get("ngram_range", [1, 2])),
        min_df=cfg.get("min_df", 2),
        max_features=cfg.get("max_features", 200_000),
        lowercase=True,
    )

    kind = cfg.get("classifier", "logreg")
    if kind == "logreg":
        clf = LogisticRegression(
            C=cfg.get("C", 4.0),
            max_iter=cfg.get("max_iter", 2000),
            class_weight=cfg.get("class_weight"),
        )
    elif kind == "svm":
        # LinearSVC gives a margin, not a probability; calibrate so the runtime threshold means
        # the same thing it does for every other family.
        clf = CalibratedClassifierCV(LinearSVC(C=cfg.get("C", 1.0)), cv=3)
    else:
        raise SystemExit(f"unknown classifier {kind!r}; expected 'logreg' or 'svm'")

    pipeline = Pipeline([("tfidf", vectorizer), ("clf", clf)])
    pipeline.fit(texts, labels)

    # training-set fit only -- a sanity check that the model learned something, not a score
    fit_acc = float(pipeline.score(texts, labels))
    print(f"training-set accuracy {fit_acc:.4f}  (fit check only, not a result)")

    export_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, export_dir / "pipeline.joblib")

    manifest = {
        "family": "tfidf",
        "name": cfg["name"],
        "version": cfg["version"],
        "vectorizer": "TfidfVectorizer",
        "classifier": kind,
        "threshold": cfg.get("threshold", 0.5),
        "train_path": cfg["train_path"],
        "train_cases": len(cases),
    }
    (export_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (run_dir / "metrics.json").write_text(
        json.dumps({**manifest, "training_set_accuracy": round(fit_acc, 4),
                    "vocabulary": len(pipeline.named_steps["tfidf"].vocabulary_)}, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"vocabulary {len(pipeline.named_steps['tfidf'].vocabulary_):,} features")
    print(f"exported {export_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
