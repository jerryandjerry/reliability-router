"""Build an embedding intent parser: encode the 9k training split, then fit cls or build knn.

The encoder is frozen — nothing is fine-tuned — so both variants are just different heads on the
same vectors. Embeddings are cached next to the run so the second variant does not pay to encode
the corpus again.

The 1k evaluation set is never read here.

Usage:
    uv run --extra embedding python models/intent_embedding/scripts/train.py \
        --config models/intent_embedding/configs/cls_v1.json
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
    export_dir = ROOT / cfg["export_dir"]
    run_dir = ROOT / cfg["output_dir"]
    variant = cfg.get("variant", "cls")

    try:
        import joblib
        import numpy as np
        from sentence_transformers import SentenceTransformer
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise SystemExit("needs sentence-transformers: uv sync --extra embedding") from exc

    cases = load_jsonl(ROOT / cfg["train_path"])
    texts = [to_text(c) for c in cases]
    labels = np.array([LABEL[c.expected_route.value] for c in cases])
    print(f"train {len(cases)} rows | DEEP {100*labels.mean():.1f}%")

    encoder_name = cfg.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
    cache = ROOT / "models" / "intent_embedding" / f"cache_{encoder_name.split('/')[-1]}.npy"

    if cache.exists():
        vectors = np.load(cache)
        if len(vectors) != len(texts):
            vectors = None
            print("cache size mismatch -> re-encoding")
        else:
            print(f"loaded cached embeddings {vectors.shape}")
    else:
        vectors = None

    if vectors is None:
        print(f"encoding with {encoder_name} ...")
        encoder = SentenceTransformer(encoder_name)
        vectors = encoder.encode(
            texts, normalize_embeddings=True, batch_size=cfg.get("batch_size", 64), show_progress_bar=True
        )
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, vectors)
        print(f"encoded {vectors.shape}, cached")

    export_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    metrics = {"train_cases": len(cases), "dimensions": int(vectors.shape[1])}

    if variant == "cls":
        # Embeddings are 384 dense dimensions -- low-dimensional with real feature interactions,
        # which is where boosted trees beat a linear head. (On TF-IDF's 200k sparse features the
        # opposite holds, which is why that family stays linear.)
        kind = cfg.get("classifier", "hgb")
        if kind == "hgb":
            clf = HistGradientBoostingClassifier(
                max_iter=cfg.get("max_iter", 500),
                learning_rate=cfg.get("learning_rate", 0.06),
                max_leaf_nodes=cfg.get("max_leaf_nodes", 31),
                min_samples_leaf=cfg.get("min_samples_leaf", 20),
                l2_regularization=cfg.get("l2", 1.0),
                early_stopping=False,
                random_state=cfg.get("seed", 20260803),
            )
        elif kind == "logreg":
            clf = LogisticRegression(C=cfg.get("C", 4.0), max_iter=cfg.get("max_iter", 2000))
        else:
            raise SystemExit(f"unknown classifier {kind!r}; expected 'hgb' or 'logreg'")
        clf.fit(vectors, labels)
        fit_acc = float(clf.score(vectors, labels))
        joblib.dump(clf, export_dir / "classifier.joblib")
        metrics["classifier"] = kind
        metrics["training_set_accuracy"] = round(fit_acc, 4)
        print(f"classifier {kind} | training-set accuracy {fit_acc:.4f}  (fit check only)")
    elif variant == "knn":
        np.savez_compressed(export_dir / "index.npz", vectors=vectors.astype("float32"), labels=labels)
        metrics["k"] = cfg.get("k", 7)
        metrics["index_rows"] = len(vectors)
        print(f"stored {len(vectors)} vectors, k={metrics['k']}")
    else:
        raise SystemExit(f"unknown variant {variant!r}; expected 'cls' or 'knn'")

    manifest = {
        "family": "embedding",
        "name": cfg["name"],
        "version": cfg["version"],
        "variant": variant,
        "embedding_model": encoder_name,
        "threshold": cfg.get("threshold", 0.5),
        "train_path": cfg["train_path"],
    }
    if variant == "knn":
        manifest["k"] = cfg.get("k", 7)
        manifest["vote"] = "distance_weighted"
    else:
        manifest["classifier"] = cfg.get("classifier", "hgb")

    (export_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps({**manifest, **metrics}, indent=2) + "\n", encoding="utf-8")
    print(f"exported {export_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
