"""Train and export each library's best configuration found by the search.

The search reports cross-validated scores; this refits each winner on the full 9,000-row training
split and writes a serving artifact, so every library becomes its own leaderboard entry rather
than a number in a log.

Artifacts land in the family folder that already has a runtime:
  tfidf      -> models/intent_tfidf/exported/<name>_v1/pipeline.joblib    (vectorizer + classifier)
  embedding  -> models/intent_embedding/exported/<name>_v1/classifier.joblib

Both runtimes call predict_proba and read `name`/`version` from the manifest, so none of them
needs new code.

Usage:
    uv run --extra gbdt python models/intent_gbdt/scripts/export_best.py --features tfidf
    uv run --extra gbdt python models/intent_gbdt/scripts/export_best.py --features embedding
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
TRAIN = ROOT / "data" / "intent_parser_splits" / "train.jsonl"
SEARCH_DIR = ROOT / "models" / "intent_gbdt" / "runs"

# the feature width the tfidf search actually used; the export must match or the scores will not
# correspond to what was selected
TFIDF_MAX_FEATURES = 50_000


def to_text(case) -> str:
    return request_to_classifier_text(
        RouterRequest(
            request_id=f"train_{case.id}",
            query=case.query,
            context=case.context,
            available_tools=case.available_tools,
        )
    )


def build_estimator(library: str, params: dict):
    if library == "hgb":
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(early_stopping=True, n_iter_no_change=20,
                                              validation_fraction=0.1, random_state=0, **params)
    if library == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(subsample_freq=1, verbosity=-1, n_jobs=2, random_state=0, **params)
    if library == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(tree_method="hist", n_jobs=2, random_state=0,
                             eval_metric="logloss", **params)
    if library == "catboost":
        from catboost import CatBoostClassifier

        return CatBoostClassifier(verbose=0, thread_count=2, random_seed=0,
                                  allow_writing_files=False, **params)
    raise SystemExit(f"unknown library {library!r}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", choices=["tfidf", "embedding"], required=True)
    ap.add_argument("--libraries", default="hgb,lightgbm,xgboost,catboost")
    args = ap.parse_args()

    search_path = SEARCH_DIR / f"search_{args.features}.json"
    if not search_path.exists():
        raise SystemExit(f"no search results at {search_path}; run search.py first")
    results = json.loads(search_path.read_text(encoding="utf-8"))

    import joblib
    import numpy as np

    cases = load_jsonl(TRAIN)
    texts = [to_text(c) for c in cases]
    y = np.array([LABEL[c.expected_route.value] for c in cases])
    print(f"train {len(cases)} rows | DEEP {100*y.mean():.1f}% | features={args.features}")

    if args.features == "embedding":
        name = "sentence-transformers/all-MiniLM-L6-v2"
        cache = ROOT / "models" / "intent_embedding" / f"cache_{name.split('/')[-1]}.npy"
        if not cache.exists():
            raise SystemExit(f"embedding cache missing: {cache}")
        X = np.load(cache)
        print(f"  reused cached embeddings {X.shape}")
        family_dir = ROOT / "models" / "intent_embedding"
    else:
        X = texts  # the vectorizer lives inside the exported pipeline
        family_dir = ROOT / "models" / "intent_tfidf"

    wanted = [x.strip() for x in args.libraries.split(",") if x.strip()]
    exported = []

    for library in wanted:
        if library not in results or not results[library].get("params"):
            print(f"\n{library}: no tuned result in {search_path.name}, skipping")
            continue

        params = results[library]["params"]
        cv = results[library]["cv_accuracy"]
        method = f"{args.features}_{library}"
        export_dir = family_dir / "exported" / f"{library}_v1"
        run_dir = family_dir.parent / "intent_gbdt" / "runs" / f"{args.features}_{library}_v1"
        print(f"\n=== {method} (search CV {cv}) ===")

        estimator = build_estimator(library, params)

        if args.features == "tfidf":
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.pipeline import Pipeline

            model = Pipeline([
                ("tfidf", TfidfVectorizer(sublinear_tf=True, strip_accents="unicode",
                                          ngram_range=(1, 2), min_df=2,
                                          max_features=TFIDF_MAX_FEATURES, lowercase=True)),
                ("clf", estimator),
            ])
            model.fit(X, y)
            artifact = "pipeline.joblib"
        else:
            model = estimator
            model.fit(X, y)
            artifact = "classifier.joblib"

        fit_acc = float(model.score(X, y))
        export_dir.mkdir(parents=True, exist_ok=True)
        run_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, export_dir / artifact)

        manifest = {
            "family": args.features,
            "name": method,
            "version": "v1",
            "library": library,
            "threshold": 0.5,
            "params": params,
            "search_cv_accuracy": cv,
            "train_path": "data/intent_parser_splits/train.jsonl",
            "train_cases": len(cases),
        }
        if args.features == "embedding":
            manifest["variant"] = "cls"
            manifest["embedding_model"] = "sentence-transformers/all-MiniLM-L6-v2"
        else:
            manifest["vectorizer"] = "TfidfVectorizer"
            manifest["max_features"] = TFIDF_MAX_FEATURES

        (export_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        (run_dir / "metrics.json").write_text(
            json.dumps({**manifest, "training_set_accuracy": round(fit_acc, 4)}, indent=2) + "\n",
            encoding="utf-8")

        print(f"  training-set accuracy {fit_acc:.4f} (fit check)")
        print(f"  exported {export_dir}")
        exported.append((method, export_dir))

    print(f"\n=== exported {len(exported)} models ===")
    parser = "embedding" if args.features == "embedding" else "tfidf"
    for _method, path in exported:
        print(f"  RR_INTENT_PARSER={parser} RR_INTENT_PARSER_MODEL={path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
