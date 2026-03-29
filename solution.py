from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path
from typing import Any

import pandas as pd

from mlebench.utils import get_logger, read_csv
from prepare import CompetitionContext, build_context, grade_submission, resolve_output_root

logger = get_logger(__name__)

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_COMPETITION_ID = "tabular-playground-series-may-2022"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the main competition solution template and emit submission/metrics artifacts."
    )
    parser.add_argument(
        "-c",
        "--competition-name",
        default=DEFAULT_COMPETITION_ID,
        help=f"Competition ID. Defaults to `{DEFAULT_COMPETITION_ID}`.",
    )
    parser.add_argument(
        "-p",
        "--path",
        type=Path,
        default=None,
        help="Prepared competition root. Defaults to `./mlebench/competitions/<competition-name>`.",
    )
    parser.add_argument(
        "--submission-path",
        type=Path,
        default=Path("submission.csv"),
        help="Path for the final submission CSV. Defaults to `./submission.csv`.",
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=Path("metrics.json"),
        help="Path for the run metrics JSON. Defaults to `./metrics.json`.",
    )
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=None,
        help="Optional analysis artifact directory to reference from metrics.",
    )
    return parser.parse_args()


def load_prepared_frames(context: CompetitionContext) -> dict[str, pd.DataFrame]:
    train_path = context.public_dir / "train.csv"
    test_path = context.public_dir / "test.csv"

    if not train_path.is_file():
        raise FileNotFoundError(f"Missing prepared train file: `{train_path}`")
    if not test_path.is_file():
        raise FileNotFoundError(f"Missing prepared test file: `{test_path}`")
    if not context.sample_submission_path.is_file():
        raise FileNotFoundError(
            f"Missing prepared sample submission file: `{context.sample_submission_path}`"
        )

    return {
        "train": read_csv(train_path),
        "test": read_csv(test_path),
        "sample_submission": read_csv(context.sample_submission_path),
    }


def infer_layout(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    sample_submission_df: pd.DataFrame,
) -> dict[str, Any]:
    id_columns = [column for column in sample_submission_df.columns if column in test_df.columns]
    if not id_columns and len(sample_submission_df.columns) > 0:
        id_columns = [sample_submission_df.columns[0]]

    prediction_columns = [
        column for column in sample_submission_df.columns if column not in id_columns
    ]
    target_columns = [column for column in train_df.columns if column not in test_df.columns]
    feature_columns = [column for column in test_df.columns if column not in id_columns]

    task_type = infer_task_type(
        train_df=train_df,
        target_columns=target_columns,
        prediction_columns=prediction_columns,
    )

    return {
        "id_columns": id_columns,
        "prediction_columns": prediction_columns,
        "target_columns": target_columns,
        "feature_columns": feature_columns,
        "task_type": task_type,
    }


def infer_task_type(
    train_df: pd.DataFrame,
    target_columns: list[str],
    prediction_columns: list[str],
) -> str:
    if not target_columns:
        return "unknown"
    if len(prediction_columns) > 1:
        return "multiclass_or_multitarget"

    target = train_df[target_columns[0]]
    non_null_target = target.dropna()

    if non_null_target.empty:
        return "unknown"
    if pd.api.types.is_bool_dtype(non_null_target):
        return "binary_classification"
    if pd.api.types.is_numeric_dtype(non_null_target):
        unique_count = int(non_null_target.nunique())
        if unique_count <= 2:
            return "binary_classification"
        if unique_count <= 20 and pd.api.types.is_integer_dtype(non_null_target):
            return "classification_or_ordinal"
        return "regression_or_probability"
    if int(non_null_target.nunique()) <= 20:
        return "classification"
    return "text_or_high_cardinality_target"


def resolve_analysis_dir(
    competition_name: str,
    analysis_dir: str | Path | None,
) -> Path:
    if analysis_dir is not None:
        return Path(analysis_dir).expanduser().resolve()
    return (REPO_ROOT / "artifacts" / competition_name / "analysis").resolve()


def load_optional_analysis_summary(analysis_dir: Path) -> dict[str, Any] | None:
    summary_path = analysis_dir / "summary.json"
    if not summary_path.is_file():
        return None
    return json.loads(summary_path.read_text(encoding="utf-8"))


def build_feature_matrices(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    layout: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # === Add preprocessing and feature engineering here.
    # === Fit transformations on train only when needed, then apply them to test.
    # === Keep submission id columns unchanged outside this feature pipeline.
    train_features = train_df[layout["feature_columns"]].copy()
    test_features = test_df[layout["feature_columns"]].copy()
    return train_features, test_features


def run_local_validation_placeholder(
    train_df: pd.DataFrame,
    train_features: pd.DataFrame,
    layout: dict[str, Any],
) -> dict[str, Any]:
    del train_df, train_features

    # === Replace this placeholder with the real local validation loop.
    # === Suggested responsibilities:
    # === 1. choose split strategy,
    # === 2. fit one or more models,
    # === 3. compute validation predictions,
    # === 4. record fold-level metrics and notes.
    return {
        "validation_scheme": "not_run_template",
        "local_score": None,
        "fold_scores": [],
        "notes": [
            "Template placeholder: validation has not been implemented yet.",
            "Use this section for CV, leakage checks, and model comparisons.",
        ],
    }


def build_placeholder_submission(
    train_df: pd.DataFrame,
    sample_submission_df: pd.DataFrame,
    layout: dict[str, Any],
) -> pd.DataFrame:
    submission_df = sample_submission_df.copy()
    prediction_columns = layout["prediction_columns"]
    target_columns = layout["target_columns"]

    # === Replace this placeholder with final-model predictions written into submission_df.
    # === Keep all submission columns and their order exactly aligned with sample_submission_df.
    if not prediction_columns:
        return submission_df

    if len(prediction_columns) == 1 and len(target_columns) == 1 and target_columns[0] in train_df.columns:
        target = train_df[target_columns[0]].dropna()
        if target.empty:
            return submission_df

        prediction_column = prediction_columns[0]
        if pd.api.types.is_numeric_dtype(target):
            submission_df[prediction_column] = float(target.mean())
        else:
            submission_df[prediction_column] = target.mode().iloc[0]
        return submission_df

    if len(prediction_columns) > 1 and len(target_columns) == 1 and target_columns[0] in train_df.columns:
        target = train_df[target_columns[0]].dropna().astype(str)
        if target.empty:
            return submission_df

        priors = target.value_counts(normalize=True)
        matched_columns = 0
        for column in prediction_columns:
            if column in priors:
                matched_columns += 1
            submission_df[column] = float(priors.get(column, 0.0))

        if matched_columns == 0:
            submission_df[prediction_columns] = 1.0 / len(prediction_columns)
        else:
            row_sums = submission_df[prediction_columns].sum(axis=1).replace(0.0, 1.0)
            submission_df[prediction_columns] = submission_df[prediction_columns].div(
                row_sums,
                axis=0,
            )

    return submission_df


def save_submission(submission_df: pd.DataFrame, submission_path: Path) -> Path:
    submission_path = submission_path.expanduser().resolve()
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    submission_df.to_csv(submission_path, index=False)
    return submission_path


def write_metrics(metrics_path: Path, metrics: dict[str, Any]) -> Path:
    metrics_path = metrics_path.expanduser().resolve()
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics_path


def run_solution(
    competition_name: str,
    path: str | Path | None = None,
    submission_path: str | Path = "submission.csv",
    metrics_path: str | Path = "metrics.json",
    analysis_dir: str | Path | None = None,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    context = build_context(competition_name, resolve_output_root(competition_name, path))
    resolved_analysis_dir = resolve_analysis_dir(competition_name, analysis_dir)
    analysis_summary = load_optional_analysis_summary(resolved_analysis_dir)

    metrics: dict[str, Any] = {
        "competition_id": competition_name,
        "status": "failed",
        "task_type": "unknown",
        "model_name": "template_placeholder",
        "validation_scheme": "not_run_template",
        "local_score": None,
        "fold_scores": [],
        "training_runtime_sec": None,
        "num_features": None,
        "submission_path": str(Path(submission_path).expanduser().resolve()),
        "metrics_path": str(Path(metrics_path).expanduser().resolve()),
        "analysis_summary_path": str((resolved_analysis_dir / "summary.json").resolve())
        if (resolved_analysis_dir / "summary.json").is_file()
        else None,
        "notes": [],
    }

    try:
        frames = load_prepared_frames(context)
        layout = infer_layout(
            train_df=frames["train"],
            test_df=frames["test"],
            sample_submission_df=frames["sample_submission"],
        )

        train_features, test_features = build_feature_matrices(
            train_df=frames["train"],
            test_df=frames["test"],
            layout=layout,
        )
        validation_result = run_local_validation_placeholder(
            train_df=frames["train"],
            train_features=train_features,
            layout=layout,
        )

        submission_df = build_placeholder_submission(
            train_df=frames["train"],
            sample_submission_df=frames["sample_submission"],
            layout=layout,
        )
        saved_submission_path = save_submission(submission_df, Path(submission_path))
        grade_result = grade_submission(
            submission_path=saved_submission_path,
            competition_name=competition_name,
            path=context.output_root,
        )

        metrics.update(
            {
                "status": "success",
                "task_type": layout["task_type"],
                "validation_scheme": validation_result["validation_scheme"],
                "local_score": validation_result["local_score"],
                "fold_scores": validation_result["fold_scores"],
                "num_features": int(test_features.shape[1]),
                "submission_path": str(saved_submission_path),
                "notes": validation_result["notes"],
                "layout": layout,
                "grading": grade_result.to_dict(),
                "analysis_summary": analysis_summary,
            }
        )

        metrics["notes"].extend(
            [
                "Template placeholder submission has been generated.",
                "Replace placeholder validation and prediction blocks with real experiments.",
            ]
        )
    except Exception as exc:
        metrics.update(
            {
                "status": "failed",
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        metrics["notes"].append("Execution failed before a full submission pipeline completed.")
    finally:
        metrics["training_runtime_sec"] = round(time.perf_counter() - started_at, 3)
        resolved_metrics_path = write_metrics(Path(metrics_path), metrics)
        metrics["metrics_path"] = str(resolved_metrics_path)

    return metrics


def main() -> None:
    args = parse_args()
    metrics = run_solution(
        competition_name=args.competition_name,
        path=args.path,
        submission_path=args.submission_path,
        metrics_path=args.metrics_path,
        analysis_dir=args.analysis_dir,
    )
    logger.info(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
