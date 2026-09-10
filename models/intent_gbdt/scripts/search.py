"""Search boosted-tree hyperparameters across libraries and feature types.

Which library and which features win is not something to reason about from priors -- it is
measured. This runs Optuna over LightGBM, XGBoost, sklearn's HistGradientBoosting and CatBoost,
on both feature representations, and reports what actually scores best.

Selection uses stratified k-fold cross-validation **inside the 9,000-row training split**. The
1,000-row evaluation set is never read here, so nothing is tuned against the numbers the
leaderboard reports.

Usage:
    uv run --extra gbdt python models/intent_gbdt/scripts/search.py \
        --features embedding --trials 60
    uv run --extra gbdt python models/intent_gbdt/scripts/search.py \
        --features tfidf --trials 60 --libraries lightgbm,xgboost
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from reliability_router.eval.runner import load_jsonl  # noqa: E402
from reliability_router.intent.text import request_to_classifier_text  # noqa: E402
from reliability_router.models import RouterRequest  # noqa: E402

LABEL = {"QUICK": 0, "DEEP": 1}
TRAIN = ROOT / "data" / "intent_parser_splits" / "train.jsonl"
OUT_DIR = ROOT / "models" / "intent_gbdt" / "runs"


def to_text(case) -> str:
    return request_to_classifier_text(
        RouterRequest(
            request_id=f"train_{case.id}",
            query=case.query,
            context=case.context,
            available_tools=case.available_tools,
        )
    )


def build_features(kind: str, texts: list[str], cfg: dict):
    """Dense sentence embeddings, or the same sparse TF-IDF the tfidf family uses."""
    import numpy as np

    if kind == "embedding":
        name = cfg.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
        cache = ROOT / "models" / "intent_embedding" / f"cache_{name.split('/')[-1]}.npy"
        if cache.exists():
            vectors = np.load(cache)
            if len(vectors) == len(texts):
                print(f"  reused cached embeddings {vectors.shape}")
                return vectors

        # Imported only on a cache miss. Loading torch initialises its own OpenMP runtime, and
        # torch + sklearn's HistGradientBoosting + LightGBM in one process segfaults on macOS.
        # With a warm cache the search never needs the encoder at all.
        from sentence_transformers import SentenceTransformer

        encoder = SentenceTransformer(name)
        vectors = encoder.encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=True)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, vectors)
        return vectors

    if kind == "tfidf":
        from sklearn.feature_extraction.text import TfidfVectorizer

        vectorizer = TfidfVectorizer(
            sublinear_tf=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            min_df=2,
            max_features=cfg.get("max_features", 200_000),
            lowercase=True,
        )
        matrix = vectorizer.fit_transform(texts)
        print(f"  TF-IDF matrix {matrix.shape}, {matrix.nnz:,} non-zeros")
        return matrix

    raise SystemExit(f"unknown feature kind {kind!r}")


def make_model(library: str, trial, sparse: bool):
    """One search space per library. Ranges are wide; the search decides, not me."""
    if library == "hgb":
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(
            max_iter=trial.suggest_int("max_iter", 100, 900),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            max_leaf_nodes=trial.suggest_int("max_leaf_nodes", 15, 127),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 5, 100),
            l2_regularization=trial.suggest_float("l2_regularization", 1e-3, 10.0, log=True),
            max_features=trial.suggest_float("max_features", 0.3, 1.0),
            early_stopping=True,
            n_iter_no_change=20,
            validation_fraction=0.1,
            random_state=0,
        )

    if library == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=trial.suggest_int("n_estimators", 100, 1500),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            num_leaves=trial.suggest_int("num_leaves", 15, 255),
            min_child_samples=trial.suggest_int("min_child_samples", 5, 100),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            subsample_freq=1,
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.3, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 20.0, log=True),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 20.0, log=True),
            verbosity=-1,
            n_jobs=2,
            random_state=0,
        )

    if library == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=trial.suggest_int("n_estimators", 100, 1500),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            max_depth=trial.suggest_int("max_depth", 3, 12),
            min_child_weight=trial.suggest_float("min_child_weight", 1e-2, 20.0, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.3, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 20.0, log=True),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 20.0, log=True),
            tree_method="hist",
            n_jobs=2,
            random_state=0,
            eval_metric="logloss",
        )

    if library == "catboost":
        from catboost import CatBoostClassifier

        return CatBoostClassifier(
            # depth and iterations are capped below the other libraries': at depth 10 with 1500
            # iterations a single trial takes ~8 minutes, 120x HGB, and the extra capacity buys
            # nothing on 9,000 rows.
            iterations=trial.suggest_int("iterations", 100, 800),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            depth=trial.suggest_int("depth", 4, 8),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
            random_strength=trial.suggest_float("random_strength", 1e-3, 10.0, log=True),
            verbose=0,
            thread_count=2,
            random_seed=0,
            allow_writing_files=False,
        )

    raise SystemExit(f"unknown library {library!r}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", choices=["embedding", "tfidf"], required=True)
    ap.add_argument("--libraries", default="hgb,lightgbm,xgboost,catboost")
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--trials-catboost", type=int, default=None,
                    help="Override the trial count for catboost, which is far slower per trial.")
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--max-features", type=int, default=200_000)
    args = ap.parse_args()

    try:
        import numpy as np
        import optuna
        from sklearn.model_selection import StratifiedKFold, cross_val_score
    except ImportError as exc:
        raise SystemExit("needs optuna and the tree libraries: uv sync --extra gbdt") from exc

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    cases = load_jsonl(TRAIN)
    texts = [to_text(c) for c in cases]
    y = np.array([LABEL[c.expected_route.value] for c in cases])
    print(f"train {len(cases)} rows | DEEP {100*y.mean():.1f}% | features={args.features}")

    X = build_features(args.features, texts, {"max_features": args.max_features})
    sparse = args.features == "tfidf"
    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=0)

    results = {}
    for library in [x.strip() for x in args.libraries.split(",") if x.strip()]:
        print(f"\n=== {library} ===")
        started = time.time()

        def objective(trial, library=library):
            model = make_model(library, trial, sparse)
            scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=1)
            return float(scores.mean())

        n_trials = args.trials_catboost if (library == "catboost" and args.trials_catboost) else args.trials
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))
        try:
            study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        except Exception as exc:  # a missing library or an unusable feature shape
            print(f"  skipped: {type(exc).__name__}: {str(exc)[:160]}")
            continue

        elapsed = time.time() - started
        results[library] = {
            "cv_accuracy": round(study.best_value, 4),
            "params": study.best_params,
            "trials": len(study.trials),
            "seconds": round(elapsed),
        }
        print(f"  best CV accuracy {study.best_value:.4f} over {len(study.trials)} trials ({elapsed:.0f}s)")
        print(f"  params {json.dumps(study.best_params)}")

    # the linear head the tree models have to beat, same folds, same features
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC

    for name, model in (("logreg", LogisticRegression(C=4.0, max_iter=2000)),
                        ("linear_svc", LinearSVC(C=1.0))):
        try:
            scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=1)
            results[name] = {"cv_accuracy": round(float(scores.mean()), 4), "params": {}, "trials": 1}
            print(f"\n{name} baseline CV accuracy {scores.mean():.4f}")
        except Exception as exc:
            print(f"\n{name} baseline failed: {type(exc).__name__}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"search_{args.features}.json"
    out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    print(f"\n=== ranking on {args.features} (cross-validated, never the eval set) ===")
    for name, r in sorted(results.items(), key=lambda kv: -kv[1]["cv_accuracy"]):
        print(f"  {name:<12} {r['cv_accuracy']:.4f}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
