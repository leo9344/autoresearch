from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
warnings.filterwarnings(
    "ignore",
    message="Could not find the number of physical cores",
    module="joblib.externals.loky.backend.context",
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="joblib.externals.loky.backend.context",
)

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from prepare import (
    DEFAULT_COMPETITION_ID,
    RANDOM_SEED,
    ROOT_DIR,
    TIME_BUDGET_SECONDS,
    compare_to_leaderboard,
    ensure_prepared_competition,
    grade_submission,
    load_gold_answers,
    read_public_data,
)


@dataclass(frozen=True)
class RunConfig:
    competition_id: str = DEFAULT_COMPETITION_ID
    time_budget_seconds: float = TIME_BUDGET_SECONDS
    search_rows: int = 50_000
    final_train_rows: int = 100_000
    n_folds: int = 2
    top_k_to_blend: int = 2


@dataclass(frozen=True)
class CandidateResult:
    name: str
    public_score: float
    fold_std: float
    fit_seconds: float
    oof_pred: np.ndarray


FEATURE_VERSION = "v1"
F27_ALPHABET = {char: idx for idx, char in enumerate("ABCDE")}


# ---------------------------------------------------------------------------
# Hyperparameters worth editing during research
# ---------------------------------------------------------------------------

SEARCH_ROWS = 50_000
FINAL_TRAIN_ROWS = 100_000
N_FOLDS = 2
TOP_K_TO_BLEND = 2

CANDIDATES = [
    (
        "logreg_l2",
        Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=0.2,
                        max_iter=500,
                        solver="lbfgs",
                    ),
                ),
            ]
        ),
    ),
    (
        "hgb_compact",
        HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=150,
            max_leaf_nodes=63,
            min_samples_leaf=200,
            l2_regularization=0.01,
            early_stopping=True,
            random_state=RANDOM_SEED,
        ),
    ),
    (
        "hgb_wide",
        HistGradientBoostingClassifier(
            learning_rate=0.03,
            max_iter=250,
            max_leaf_nodes=95,
            min_samples_leaf=120,
            l2_regularization=0.001,
            early_stopping=True,
            random_state=RANDOM_SEED + 1,
        ),
    ),
]


# ---------------------------------------------------------------------------
# Data + feature engineering
# ---------------------------------------------------------------------------


def stratified_subsample_indices(y: np.ndarray, n_rows: int | None) -> np.ndarray:
    if n_rows is None or n_rows <= 0 or n_rows >= len(y):
        return np.arange(len(y))
    splitter = StratifiedShuffleSplit(n_splits=1, train_size=n_rows, random_state=RANDOM_SEED)
    indices, _ = next(splitter.split(np.zeros(len(y)), y))
    return np.sort(indices)



def _block_stats(frame: pd.DataFrame, cols: list[str], prefix: str) -> pd.DataFrame:
    block = frame[cols].to_numpy(dtype=np.float32, copy=False)
    stats = pd.DataFrame(index=frame.index)
    stats[f"{prefix}_sum"] = block.sum(axis=1)
    stats[f"{prefix}_mean"] = block.mean(axis=1)
    stats[f"{prefix}_std"] = block.std(axis=1)
    stats[f"{prefix}_min"] = block.min(axis=1)
    stats[f"{prefix}_max"] = block.max(axis=1)
    return stats



def build_feature_matrices(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    y_train = train_df["target"].to_numpy(dtype=np.int8, copy=True)
    test_ids = test_df["id"].to_numpy(copy=True)

    combined = pd.concat(
        [train_df.drop(columns=["target"]), test_df],
        axis=0,
        ignore_index=True,
    )

    base = combined.drop(columns=["id", "f_27"]).copy()
    base = base.astype(np.float32)

    early_float_cols = [f"f_{i:02d}" for i in range(0, 7)]
    int_block_cols = [f"f_{i:02d}" for i in range(7, 19)]
    late_float_cols = [f"f_{i:02d}" for i in range(19, 27)] + ["f_28"]
    binary_cols = ["f_29", "f_30"]

    engineered = [
        base,
        _block_stats(base, early_float_cols, "early_float"),
        _block_stats(base, int_block_cols, "int_block"),
        _block_stats(base, late_float_cols, "late_float"),
        _block_stats(base, binary_cols, "binary_block"),
    ]

    engineered.append(
        pd.DataFrame(
            {
                "f_00_x_f_19": base["f_00"] * base["f_19"],
                "f_01_x_f_21": base["f_01"] * base["f_21"],
                "f_02_x_f_28": base["f_02"] * base["f_28"],
                "f_05_minus_f_26": base["f_05"] - base["f_26"],
                "int_vs_float_gap": base[int_block_cols].mean(axis=1) - base[late_float_cols].mean(axis=1),
            },
            index=base.index,
        )
    )

    f27 = combined["f_27"].fillna("").astype(str).str.pad(10, side="right", fillchar="A").str.slice(0, 10)
    char_cols = {}
    for pos in range(10):
        char_cols[f"f_27_{pos:02d}"] = (
            f27.str[pos].map(F27_ALPHABET).fillna(-1).astype(np.int8)
        )
    f27_frame = pd.DataFrame(char_cols, index=combined.index)
    f27_values = f27_frame.to_numpy(dtype=np.int8, copy=False)

    f27_counts = {
        f"f_27_count_{char}": (f27_values == code).sum(axis=1).astype(np.int8)
        for char, code in F27_ALPHABET.items()
    }
    f27_extra = pd.DataFrame(f27_counts, index=combined.index)
    f27_extra["f_27_unique"] = f27_frame.nunique(axis=1).astype(np.int8)
    f27_extra["f_27_transitions"] = (f27_values[:, 1:] != f27_values[:, :-1]).sum(axis=1).astype(np.int8)
    f27_extra["f_27_repeat_pairs"] = (f27_values[:, 1:] == f27_values[:, :-1]).sum(axis=1).astype(np.int8)

    engineered.extend([f27_frame.astype(np.float32), f27_extra.astype(np.float32)])
    feature_frame = pd.concat(engineered, axis=1)
    feature_frame = feature_frame.astype(np.float32)

    n_train = len(train_df)
    x_train = feature_frame.iloc[:n_train].to_numpy(dtype=np.float32, copy=False)
    x_test = feature_frame.iloc[n_train:].to_numpy(dtype=np.float32, copy=False)
    return x_train, y_train, x_test, test_ids, feature_frame.columns.to_list()


# ---------------------------------------------------------------------------
# Model evaluation + fitting
# ---------------------------------------------------------------------------


def predict_positive_proba(model, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.clip(model.predict_proba(x)[:, 1], 1e-6, 1 - 1e-6)
    raise TypeError(f"Model `{type(model).__name__}` does not expose predict_proba().")



def evaluate_candidate(name: str, estimator, x: np.ndarray, y: np.ndarray, n_folds: int) -> CandidateResult:
    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    oof_pred = np.zeros(len(y), dtype=np.float32)
    fold_scores: list[float] = []
    t0 = time.time()

    for fold_idx, (train_idx, valid_idx) in enumerate(splitter.split(x, y), start=1):
        model = clone(estimator)
        model.fit(x[train_idx], y[train_idx])
        valid_pred = predict_positive_proba(model, x[valid_idx])
        oof_pred[valid_idx] = valid_pred
        fold_score = float(roc_auc_score(y[valid_idx], valid_pred))
        fold_scores.append(fold_score)
        print(f"candidate {name:>12s} | fold {fold_idx}/{n_folds} | auc={fold_score:.6f}")

    return CandidateResult(
        name=name,
        public_score=float(roc_auc_score(y, oof_pred)),
        fold_std=float(np.std(fold_scores)),
        fit_seconds=time.time() - t0,
        oof_pred=oof_pred,
    )



def blend_oof_predictions(results: list[CandidateResult]) -> tuple[np.ndarray, np.ndarray]:
    if len(results) == 1:
        return results[0].oof_pred.copy(), np.array([1.0], dtype=np.float32)
    raw_weights = np.array([max(result.public_score - 0.5, 1e-6) for result in results], dtype=np.float32)
    weights = raw_weights / raw_weights.sum()
    blended = np.zeros_like(results[0].oof_pred)
    for weight, result in zip(weights, results):
        blended += weight * result.oof_pred
    return np.clip(blended, 1e-6, 1 - 1e-6), weights



def fit_and_predict(estimator, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    model = clone(estimator)
    model.fit(x_train, y_train)
    return predict_positive_proba(model, x_test)


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(description="Single-file AutoML experiment script.")
    parser.add_argument("--competition-id", default=DEFAULT_COMPETITION_ID)
    parser.add_argument("--time-budget", type=float, default=TIME_BUDGET_SECONDS)
    parser.add_argument("--search-rows", type=int, default=SEARCH_ROWS)
    parser.add_argument("--final-train-rows", type=int, default=FINAL_TRAIN_ROWS)
    parser.add_argument("--folds", type=int, default=N_FOLDS)
    parser.add_argument("--top-k", type=int, default=TOP_K_TO_BLEND)
    args = parser.parse_args()
    return RunConfig(
        competition_id=args.competition_id,
        time_budget_seconds=args.time_budget,
        search_rows=args.search_rows,
        final_train_rows=args.final_train_rows,
        n_folds=args.folds,
        top_k_to_blend=args.top_k,
    )



def main() -> None:
    config = parse_args()
    run_started = time.time()
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    paths = ensure_prepared_competition(config.competition_id)
    train_df, test_df = read_public_data(paths)
    gold_answers = load_gold_answers(paths)

    x_train, y_train, x_test, test_ids, feature_names = build_feature_matrices(train_df, test_df)
    search_indices = stratified_subsample_indices(y_train, config.search_rows)
    x_search = x_train[search_indices]
    y_search = y_train[search_indices]

    print(f"Competition: {paths.competition_id}")
    print(f"Feature version: {FEATURE_VERSION}")
    print(f"Train rows: {len(y_train):,} | Search rows: {len(y_search):,} | Features: {len(feature_names)}")

    candidate_results: list[CandidateResult] = []
    for name, estimator in CANDIDATES:
        elapsed = time.time() - run_started
        if candidate_results and elapsed >= config.time_budget_seconds * 0.85:
            print(f"Skipping remaining candidates because {elapsed:.1f}s of the {config.time_budget_seconds:.1f}s budget is already used.")
            break
        try:
            result = evaluate_candidate(name, estimator, x_search, y_search, config.n_folds)
            candidate_results.append(result)
            print(
                f"candidate {name:>12s} | public_score={result.public_score:.6f} | "
                f"fold_std={result.fold_std:.6f} | seconds={result.fit_seconds:.1f}"
            )
        except Exception as exc:
            print(f"candidate {name:>12s} failed: {type(exc).__name__}: {exc}")

    if candidate_results:
        candidate_results.sort(key=lambda item: item.public_score, reverse=True)
        selected = candidate_results[: max(1, min(config.top_k_to_blend, len(candidate_results)))]
        blended_oof, weights = blend_oof_predictions(selected)
        public_score = float(roc_auc_score(y_search, blended_oof))
    else:
        selected = []
        weights = np.array([1.0], dtype=np.float32)
        public_score = 0.5

    elapsed_after_search = time.time() - run_started
    remaining_time = config.time_budget_seconds - elapsed_after_search

    if config.final_train_rows is None or config.final_train_rows < 0:
        final_indices = np.arange(len(y_train))
    else:
        final_indices = stratified_subsample_indices(y_train, config.final_train_rows)

    if selected and remaining_time < 30 and len(final_indices) > len(search_indices):
        final_indices = search_indices
    if selected and remaining_time < 20:
        selected = selected[:1]
        weights = np.array([1.0], dtype=np.float32)

    x_final = x_train[final_indices]
    y_final = y_train[final_indices]

    if selected:
        predictions = np.zeros(len(test_ids), dtype=np.float32)
        selected_lookup = {name: estimator for name, estimator in CANDIDATES}
        for weight, result in zip(weights, selected):
            test_pred = fit_and_predict(selected_lookup[result.name], x_final, y_final, x_test)
            predictions += weight * test_pred
        predictions = np.clip(predictions, 1e-6, 1 - 1e-6)
        selected_model_names = [result.name for result in selected]
    else:
        positive_rate = float(y_final.mean()) if len(y_final) else float(y_train.mean())
        predictions = np.full(len(test_ids), positive_rate, dtype=np.float32)
        selected_model_names = ["constant_mean"]

    submission = pd.DataFrame({"id": test_ids, "target": predictions})
    private_score = float(grade_submission(submission, gold_answers))
    private_report = compare_to_leaderboard(private_score, paths.leaderboard_path)

    submission_path = paths.submissions_dir / f"{timestamp}_submission.csv"
    latest_submission_path = ROOT_DIR / "submission.csv"
    submission.to_csv(submission_path, index=False)
    submission.to_csv(latest_submission_path, index=False)

    candidate_table = [
        {
            "name": result.name,
            "public_score": result.public_score,
            "fold_std": result.fold_std,
            "fit_seconds": result.fit_seconds,
        }
        for result in candidate_results
    ]

    run_summary = {
        "timestamp": timestamp,
        "competition_id": paths.competition_id,
        "feature_version": FEATURE_VERSION,
        "config": asdict(config),
        "selected_models": selected_model_names,
        "selected_weights": weights.tolist(),
        "candidate_results": candidate_table,
        "train_rows": int(len(y_train)),
        "search_rows": int(len(y_search)),
        "final_train_rows": int(len(final_indices)),
        "num_features": int(len(feature_names)),
        "public_score": public_score,
        "private_score": private_report.score,
        "medal": private_report.medal,
        "estimated_rank": private_report.estimated_rank,
        "total_teams": private_report.total_teams,
        "gold_threshold": private_report.gold_threshold,
        "silver_threshold": private_report.silver_threshold,
        "bronze_threshold": private_report.bronze_threshold,
        "median_threshold": private_report.median_threshold,
        "submission_path": str(submission_path),
        "latest_submission_path": str(latest_submission_path),
        "train_seconds": time.time() - run_started,
    }

    paths.runs_dir.mkdir(parents=True, exist_ok=True)
    run_summary_path = paths.runs_dir / f"{timestamp}_run.json"
    latest_run_path = paths.artifacts_dir / "latest_run.json"
    run_summary_path.write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
    latest_run_path.write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
    pd.DataFrame(candidate_table).to_csv(paths.artifacts_dir / "latest_candidates.csv", index=False)

    print("---")
    print(f"competition_id:    {paths.competition_id}")
    print(f"feature_version:   {FEATURE_VERSION}")
    print(f"public_score:      {public_score:.6f}")
    print(f"private_score:     {private_report.score:.6f}")
    print(f"medal:             {private_report.medal}")
    print(f"estimated_rank:    {private_report.estimated_rank}")
    print(f"gold_threshold:    {private_report.gold_threshold}")
    print(f"silver_threshold:  {private_report.silver_threshold}")
    print(f"bronze_threshold:  {private_report.bronze_threshold}")
    print(f"median_threshold:  {private_report.median_threshold}")
    print(f"train_seconds:     {run_summary['train_seconds']:.1f}")
    print(f"search_rows:       {len(y_search)}")
    print(f"final_train_rows:  {len(final_indices)}")
    print(f"num_features:      {len(feature_names)}")
    print(f"selected_models:   {', '.join(selected_model_names)}")
    print(f"submission_path:   {submission_path}")
    print(f"run_summary_path:  {run_summary_path}")


if __name__ == "__main__":
    main()
