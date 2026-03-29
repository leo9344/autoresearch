from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
import traceback
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from mlebench.utils import get_logger, read_csv
from prepare import CompetitionContext, build_context, grade_submission, resolve_output_root

logger = get_logger(__name__)

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_COMPETITION_ID = "tabular-playground-series-may-2022"
RESULTS_HEADER = ["commit", "public_score", "private_score", "status", "description"]


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
    parser.add_argument(
        "--results-path",
        type=Path,
        default=Path("results.tsv"),
        help="Path for the tab-separated experiment log. Defaults to `./results.tsv`.",
    )
    parser.add_argument(
        "--experiment-description",
        default=None,
        help="Optional short description to store in `results.tsv` for this run.",
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


def apply_placeholder_predictions(
    reference_train_df: pd.DataFrame,
    submission_df: pd.DataFrame,
    layout: dict[str, Any],
) -> pd.DataFrame:
    submission_df = submission_df.copy()
    prediction_columns = layout["prediction_columns"]
    target_columns = layout["target_columns"]

    if not prediction_columns:
        return submission_df

    if len(prediction_columns) == 1 and len(target_columns) == 1 and target_columns[0] in reference_train_df.columns:
        target = reference_train_df[target_columns[0]].dropna()
        if target.empty:
            return submission_df

        prediction_column = prediction_columns[0]
        if pd.api.types.is_numeric_dtype(target):
            submission_df[prediction_column] = float(target.mean())
        else:
            submission_df[prediction_column] = target.mode().iloc[0]
        return submission_df

    if len(prediction_columns) > 1 and len(target_columns) == 1 and target_columns[0] in reference_train_df.columns:
        target = reference_train_df[target_columns[0]].dropna().astype(str)
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


def build_validation_submission_template(
    validation_df: pd.DataFrame,
    sample_submission_df: pd.DataFrame,
    layout: dict[str, Any],
) -> pd.DataFrame:
    template = pd.DataFrame(index=validation_df.index)

    for column in sample_submission_df.columns:
        if column in layout["id_columns"] and column in validation_df.columns:
            template[column] = validation_df[column].to_numpy()
        elif column in layout["prediction_columns"]:
            template[column] = 0.0
        else:
            template[column] = None

    return template.reset_index(drop=True)[sample_submission_df.columns]


def build_validation_answers(
    validation_df: pd.DataFrame,
    layout: dict[str, Any],
) -> pd.DataFrame:
    answer_columns = list(dict.fromkeys(layout["id_columns"] + layout["target_columns"]))
    return validation_df[answer_columns].reset_index(drop=True).copy()


def choose_validation_stratify(
    train_df: pd.DataFrame,
    layout: dict[str, Any],
) -> pd.Series | None:
    if len(layout["target_columns"]) != 1:
        return None
    if layout["task_type"] not in {
        "binary_classification",
        "classification",
        "classification_or_ordinal",
    }:
        return None

    target = train_df[layout["target_columns"][0]]
    if target.isna().any():
        return None

    value_counts = target.value_counts(dropna=True)
    if len(value_counts) <= 1:
        return None
    if int(value_counts.min()) < 2:
        return None
    return target


def run_public_validation_placeholder(
    context: CompetitionContext,
    train_df: pd.DataFrame,
    train_features: pd.DataFrame,
    sample_submission_df: pd.DataFrame,
    layout: dict[str, Any],
) -> dict[str, Any]:
    del train_features

    # === Replace this placeholder with the real local/public validation loop.
    # === Suggested responsibilities:
    # === 1. choose split strategy,
    # === 2. fit one or more models,
    # === 3. compute validation predictions,
    # === 4. record fold-level metrics and notes.
    if not layout["target_columns"]:
        return {
            "validation_scheme": "not_run_template",
            "public_score": None,
            "fold_scores": [],
            "notes": [
                "Template placeholder: no target columns were inferred, so public validation was skipped.",
            ],
        }

    stratify = choose_validation_stratify(train_df, layout)
    train_split, validation_split = train_test_split(
        train_df,
        test_size=0.2,
        random_state=0,
        stratify=stratify,
    )

    validation_submission = build_validation_submission_template(
        validation_df=validation_split,
        sample_submission_df=sample_submission_df,
        layout=layout,
    )
    validation_submission = apply_placeholder_predictions(
        reference_train_df=train_split,
        submission_df=validation_submission,
        layout=layout,
    )
    validation_answers = build_validation_answers(validation_split, layout)
    public_score = context.grader(validation_submission, validation_answers)

    notes = [
        "Template placeholder: public_score comes from a simple 80/20 holdout baseline.",
        "Replace this section with a real CV strategy aligned to the competition metric.",
    ]
    if public_score is None:
        notes.append("Public validation could not be scored with the current placeholder baseline.")
    else:
        notes.append("The placeholder public validation uses the competition grader directly.")

    return {
        "validation_scheme": "public_holdout_placeholder",
        "public_score": float(public_score) if public_score is not None else None,
        "fold_scores": [float(public_score)] if public_score is not None else [],
        "notes": notes,
    }


def build_placeholder_submission(
    train_df: pd.DataFrame,
    sample_submission_df: pd.DataFrame,
    layout: dict[str, Any],
) -> pd.DataFrame:
    # === Replace this placeholder with final-model predictions written into submission_df.
    # === Keep all submission columns and their order exactly aligned with sample_submission_df.
    return apply_placeholder_predictions(
        reference_train_df=train_df,
        submission_df=sample_submission_df.copy(),
        layout=layout,
    )


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


def sanitize_tsv_cell(value: str | None) -> str:
    if value is None:
        return ""
    return str(value).replace("\t", " ").replace("\r", " ").replace("\n", " ").strip()


def format_score(score: float | None) -> str:
    if score is None:
        return ""
    return f"{float(score):.6f}"


def run_git_command(*args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None

    output = completed.stdout.strip()
    return output or None


def infer_is_lower_better(metric_name: str | None) -> bool | None:
    if not metric_name:
        return None

    metric_name = metric_name.lower()
    lower_better_tokens = ("loss", "rmse", "mse", "mae", "error", "perplexity", "bpb")
    higher_better_tokens = ("auc", "accuracy", "f1", "precision", "recall", "r2", "map", "ndcg")

    if any(token in metric_name for token in lower_better_tokens):
        return True
    if any(token in metric_name for token in higher_better_tokens):
        return False
    return None


def resolve_score_direction(grading: dict[str, Any] | None) -> tuple[bool | None, str]:
    if grading is None:
        return None, "missing"

    explicit_direction = grading.get("is_lower_better")
    if explicit_direction is not None:
        return bool(explicit_direction), "grading.is_lower_better"

    inferred_direction = infer_is_lower_better(grading.get("metric_name"))
    if inferred_direction is not None:
        return inferred_direction, "metric_name_heuristic"

    return None, "unknown"


def parse_logged_score(value: str | None) -> float | None:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    try:
        return float(stripped)
    except ValueError:
        return None


def score_is_better(candidate: float, incumbent: float | None, is_lower_better: bool) -> bool:
    if incumbent is None:
        return True
    if is_lower_better:
        return candidate < incumbent
    return candidate > incumbent


def load_previous_best_private_score(
    results_path: Path,
    is_lower_better: bool | None,
) -> float | None:
    if not results_path.is_file() or is_lower_better is None:
        return None

    best_score: float | None = None
    with results_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if row.get("status") != "keep":
                continue
            score = parse_logged_score(row.get("private_score"))
            if score is None:
                continue
            if score_is_better(score, best_score, is_lower_better):
                best_score = score

    return best_score


def determine_results_status(
    metrics: dict[str, Any],
    previous_best_private_score: float | None,
    is_lower_better: bool | None,
) -> str:
    if metrics.get("status") != "success":
        return "crash"

    grading = metrics.get("grading") or {}
    private_score = metrics.get("private_score")
    valid_submission = bool(grading.get("valid_submission"))

    if not valid_submission or private_score is None:
        return "crash"
    if previous_best_private_score is None:
        return "keep"
    if is_lower_better is None:
        return "keep"
    if score_is_better(float(private_score), previous_best_private_score, is_lower_better):
        return "keep"
    return "discard"


def resolve_experiment_description(
    metrics: dict[str, Any],
    experiment_description: str | None,
) -> str:
    if experiment_description and experiment_description.strip():
        return sanitize_tsv_cell(experiment_description)

    notes = metrics.get("notes") or []
    for note in notes:
        cleaned = sanitize_tsv_cell(str(note))
        if cleaned:
            return cleaned

    error_message = metrics.get("error_message")
    if error_message:
        return sanitize_tsv_cell(str(error_message))

    return "run"


def append_results_row(
    results_path: Path,
    metrics: dict[str, Any],
    experiment_description: str | None,
) -> dict[str, Any]:
    results_path = results_path.expanduser().resolve()
    results_path.parent.mkdir(parents=True, exist_ok=True)

    grading = metrics.get("grading") or {}
    is_lower_better, direction_source = resolve_score_direction(grading)
    previous_best_private_score = load_previous_best_private_score(results_path, is_lower_better)
    status = determine_results_status(metrics, previous_best_private_score, is_lower_better)

    row = {
        "commit": sanitize_tsv_cell(run_git_command("rev-parse", "--short", "HEAD") or "unknown"),
        "public_score": format_score(metrics.get("public_score")),
        "private_score": format_score(metrics.get("private_score")),
        "status": status,
        "description": resolve_experiment_description(metrics, experiment_description),
    }

    should_write_header = not results_path.is_file() or results_path.stat().st_size == 0
    with results_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=RESULTS_HEADER,
            delimiter="\t",
            extrasaction="ignore",
        )
        if should_write_header:
            writer.writeheader()
        writer.writerow(row)

    return {
        "path": str(results_path),
        "row": row,
        "direction_source": direction_source,
        "previous_best_private_score": previous_best_private_score,
    }


def run_solution(
    competition_name: str,
    path: str | Path | None = None,
    submission_path: str | Path = "submission.csv",
    metrics_path: str | Path = "metrics.json",
    analysis_dir: str | Path | None = None,
    results_path: str | Path = "results.tsv",
    experiment_description: str | None = None,
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
        "metric_name": None,
        "validation_scheme": "not_run_template",
        "public_score": None,
        "private_score": None,
        "valid_submission": False,
        "fold_scores": [],
        "training_runtime_sec": None,
        "num_features": None,
        "submission_path": str(Path(submission_path).expanduser().resolve()),
        "metrics_path": str(Path(metrics_path).expanduser().resolve()),
        "results_path": str(Path(results_path).expanduser().resolve()),
        "analysis_summary_path": str((resolved_analysis_dir / "summary.json").resolve())
        if (resolved_analysis_dir / "summary.json").is_file()
        else None,
        "git_commit": run_git_command("rev-parse", "--short", "HEAD"),
        "git_branch": run_git_command("branch", "--show-current"),
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
        validation_result = run_public_validation_placeholder(
            context=context,
            train_df=frames["train"],
            train_features=train_features,
            sample_submission_df=frames["sample_submission"],
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
                "metric_name": grade_result.metric_name,
                "validation_scheme": validation_result["validation_scheme"],
                "public_score": validation_result["public_score"],
                "private_score": grade_result.score,
                "valid_submission": grade_result.valid_submission,
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
                "Replace placeholder public validation and prediction blocks with real experiments.",
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
        metrics["results_log"] = append_results_row(
            results_path=Path(results_path),
            metrics=metrics,
            experiment_description=experiment_description,
        )
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
        results_path=args.results_path,
        experiment_description=args.experiment_description,
    )
    logger.info(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
