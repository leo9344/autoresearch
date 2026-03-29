from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

from mlebench.utils import get_logger, load_answers, read_csv
from prepare import CompetitionContext, build_context, resolve_output_root

logger = get_logger(__name__)

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_COMPETITION_ID = "tabular-playground-series-may-2022"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Produce structured analysis artifacts for one prepared competition."
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
        "--output-dir",
        type=Path,
        default=None,
        help="Analysis artifact directory. Defaults to `./artifacts/<competition-name>/analysis`.",
    )
    return parser.parse_args()


def resolve_analysis_output_dir(
    competition_name: str,
    output_dir: str | Path | None,
) -> Path:
    if output_dir is not None:
        return Path(output_dir).expanduser().resolve()
    return (REPO_ROOT / "artifacts" / competition_name / "analysis").resolve()


def load_prepared_frames(context: CompetitionContext) -> dict[str, Any]:
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

    frames: dict[str, Any] = {
        "train": read_csv(train_path),
        "test": read_csv(test_path),
        "sample_submission": read_csv(context.sample_submission_path),
    }

    if context.answers_path.is_file():
        answers = load_answers(context.answers_path)
        if isinstance(answers, pd.DataFrame):
            frames["answers"] = answers
        else:
            frames["answers"] = None
            frames["answers_kind"] = type(answers).__name__
    else:
        frames["answers"] = None
        frames["answers_kind"] = None

    return frames


def infer_layout(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    sample_submission_df: pd.DataFrame,
    answers_df: pd.DataFrame | None,
) -> dict[str, Any]:
    id_columns = [column for column in sample_submission_df.columns if column in test_df.columns]
    if not id_columns and len(sample_submission_df.columns) > 0:
        id_columns = [sample_submission_df.columns[0]]

    prediction_columns = [
        column for column in sample_submission_df.columns if column not in id_columns
    ]

    train_only_columns = [column for column in train_df.columns if column not in test_df.columns]
    answer_only_columns: list[str] = []
    if answers_df is not None:
        answer_only_columns = [
            column for column in answers_df.columns if column not in test_df.columns
        ]

    target_columns = answer_only_columns or train_only_columns
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


def summarize_columns(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    layout: dict[str, Any],
) -> list[dict[str, Any]]:
    all_columns = list(dict.fromkeys(list(train_df.columns) + list(test_df.columns)))
    summary_rows: list[dict[str, Any]] = []

    for column in all_columns:
        in_train = column in train_df.columns
        in_test = column in test_df.columns
        train_series = train_df[column] if in_train else None
        test_series = test_df[column] if in_test else None

        role = "feature"
        if column in layout["id_columns"]:
            role = "id"
        elif column in layout["target_columns"]:
            role = "target"
        elif in_train and not in_test:
            role = "train_only"
        elif in_test and not in_train:
            role = "test_only"

        summary_rows.append(
            {
                "column": column,
                "role": role,
                "in_train": in_train,
                "in_test": in_test,
                "train_dtype": str(train_series.dtype) if train_series is not None else None,
                "test_dtype": str(test_series.dtype) if test_series is not None else None,
                "train_missing_count": int(train_series.isna().sum()) if train_series is not None else None,
                "test_missing_count": int(test_series.isna().sum()) if test_series is not None else None,
                "train_missing_rate": round(float(train_series.isna().mean()), 6)
                if train_series is not None
                else None,
                "test_missing_rate": round(float(test_series.isna().mean()), 6)
                if test_series is not None
                else None,
                "train_nunique": int(train_series.nunique(dropna=True)) if train_series is not None else None,
                "test_nunique": int(test_series.nunique(dropna=True)) if test_series is not None else None,
            }
        )

    return summary_rows


def collect_analysis_summary(
    context: CompetitionContext,
    frames: dict[str, Any],
    layout: dict[str, Any],
    column_rows: list[dict[str, Any]],
    runtime_sec: float,
) -> dict[str, Any]:
    train_df = frames["train"]
    test_df = frames["test"]
    answers_df = frames.get("answers")

    train_missing = [
        {
            "column": row["column"],
            "missing_rate": row["train_missing_rate"],
        }
        for row in column_rows
        if row["train_missing_rate"] not in (None, 0.0)
    ]
    train_missing = sorted(train_missing, key=lambda item: item["missing_rate"], reverse=True)[:20]

    categorical_rows = [
        {
            "column": row["column"],
            "train_nunique": row["train_nunique"],
        }
        for row in column_rows
        if row["role"] == "feature"
        and row["train_dtype"] is not None
        and "object" in row["train_dtype"]
    ]
    categorical_rows = sorted(
        categorical_rows,
        key=lambda item: item["train_nunique"] if item["train_nunique"] is not None else -1,
        reverse=True,
    )[:20]

    suspicious_id_like_columns = []
    for row in column_rows:
        unique_count = row["train_nunique"]
        if unique_count is None:
            continue
        if row["column"].lower().endswith("id") or row["column"].lower() == "id":
            suspicious_id_like_columns.append(row["column"])
            continue
        if row["role"] == "feature" and unique_count >= max(1, int(0.98 * len(train_df))):
            suspicious_id_like_columns.append(row["column"])

    summary = {
        "competition_id": context.competition_id,
        "data_root": str(context.output_root),
        "task_type": layout["task_type"],
        "train_shape": [int(train_df.shape[0]), int(train_df.shape[1])],
        "test_shape": [int(test_df.shape[0]), int(test_df.shape[1])],
        "answers_loaded": answers_df is not None,
        "answers_path": str(context.answers_path),
        "sample_submission_path": str(context.sample_submission_path),
        "id_columns": layout["id_columns"],
        "target_columns": layout["target_columns"],
        "prediction_columns": layout["prediction_columns"],
        "feature_columns": layout["feature_columns"],
        "num_features": int(len(layout["feature_columns"])),
        "train_missing_top_columns": train_missing,
        "high_cardinality_categorical_columns": categorical_rows,
        "suspicious_id_like_columns": suspicious_id_like_columns,
        "validation_hints": suggest_validation_hints(train_df, layout),
        "model_hints": suggest_model_hints(column_rows, layout),
        "runtime_sec": round(runtime_sec, 3),
    }

    if len(layout["target_columns"]) == 1 and layout["target_columns"][0] in train_df.columns:
        target = train_df[layout["target_columns"][0]]
        summary["target_summary"] = summarize_target(target)

    return summary


def summarize_target(target: pd.Series) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "dtype": str(target.dtype),
        "missing_rate": round(float(target.isna().mean()), 6),
        "nunique": int(target.nunique(dropna=True)),
    }

    non_null_target = target.dropna()
    if non_null_target.empty:
        return summary

    if pd.api.types.is_numeric_dtype(non_null_target):
        summary.update(
            {
                "mean": float(non_null_target.mean()),
                "std": float(non_null_target.std()) if len(non_null_target) > 1 else 0.0,
                "min": float(non_null_target.min()),
                "max": float(non_null_target.max()),
            }
        )
    else:
        value_counts = non_null_target.astype(str).value_counts(normalize=True).head(10)
        summary["top_values"] = {
            str(label): round(float(value), 6) for label, value in value_counts.items()
        }

    return summary


def suggest_validation_hints(train_df: pd.DataFrame, layout: dict[str, Any]) -> list[str]:
    hints: list[str] = []
    lower_columns = {column.lower(): column for column in train_df.columns}

    if any(token in column_name for column_name in lower_columns for token in ("date", "time", "timestamp")):
        hints.append("Prefer a time-aware split if any timestamp-like column is truly temporal.")

    if layout["task_type"] in {"binary_classification", "classification"}:
        hints.append("Consider stratified validation if target imbalance is nontrivial.")
    else:
        hints.append("Start with a simple train/validation split or K-fold baseline.")

    if layout["id_columns"]:
        hints.append("Exclude obvious id columns from feature sets unless there is evidence they carry signal.")

    return hints


def suggest_model_hints(column_rows: list[dict[str, Any]], layout: dict[str, Any]) -> list[str]:
    hints: list[str] = []
    has_missing = any(
        row["role"] == "feature" and (row["train_missing_rate"] or 0.0) > 0.0 for row in column_rows
    )
    has_object_features = any(
        row["role"] == "feature" and row["train_dtype"] is not None and "object" in row["train_dtype"]
        for row in column_rows
    )

    if has_object_features:
        hints.append("CatBoost or other tree models with strong categorical handling are natural first baselines.")
    else:
        hints.append("Tree ensembles are a safe baseline for mostly numeric tabular data.")

    if has_missing:
        hints.append("Missingness indicators and native missing-value handling are worth testing early.")

    if layout["task_type"] == "regression_or_probability":
        hints.append("Start with a regression baseline, then compare boosted trees against linear models if needed.")
    elif layout["task_type"] in {"binary_classification", "classification"}:
        hints.append("Start with a classification baseline aligned to the competition metric.")

    return hints


def build_notes_markdown(summary: dict[str, Any]) -> str:
    lines = [
        f"# Analysis Notes: {summary['competition_id']}",
        "",
        "## Confirmed Inputs",
        f"- data_root: `{summary['data_root']}`",
        f"- task_type: `{summary['task_type']}`",
        f"- id_columns: `{summary['id_columns']}`",
        f"- target_columns: `{summary['target_columns']}`",
        f"- prediction_columns: `{summary['prediction_columns']}`",
        "",
        "## Quick Findings",
        f"- train_shape: `{summary['train_shape']}`",
        f"- test_shape: `{summary['test_shape']}`",
        f"- num_features: `{summary['num_features']}`",
        f"- suspicious_id_like_columns: `{summary['suspicious_id_like_columns']}`",
        "",
        "## Recommended Next Checks",
    ]

    for hint in summary["validation_hints"] + summary["model_hints"]:
        lines.append(f"- {hint}")

    lines.extend(
        [
            "",
            "## === Leakage Review",
            "- === Check whether any train-only column leaks target information.",
            "",
            "## === Shift Review",
            "- === Compare train/test distributions for high-impact numeric and categorical columns.",
            "",
            "## === Feature Ideas",
            "- === Add candidate feature transformations backed by the current analysis.",
            "",
            "## === Modeling Notes",
            "- === Record which baseline families are most promising for the next `solution.py` iteration.",
            "",
        ]
    )

    return "\n".join(lines)


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def run_analysis(
    competition_name: str,
    path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    started_at = time.perf_counter()

    context = build_context(competition_name, resolve_output_root(competition_name, path))
    frames = load_prepared_frames(context)
    layout = infer_layout(
        train_df=frames["train"],
        test_df=frames["test"],
        sample_submission_df=frames["sample_submission"],
        answers_df=frames.get("answers"),
    )

    output_dir_path = resolve_analysis_output_dir(competition_name, output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    column_rows = summarize_columns(frames["train"], frames["test"], layout)
    column_report_path = output_dir_path / "column_report.csv"
    pd.DataFrame(column_rows).to_csv(column_report_path, index=False)

    runtime_sec = time.perf_counter() - started_at
    summary = collect_analysis_summary(
        context=context,
        frames=frames,
        layout=layout,
        column_rows=column_rows,
        runtime_sec=runtime_sec,
    )

    summary_path = output_dir_path / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    notes_path = output_dir_path / "notes.md"
    notes_path.write_text(build_notes_markdown(summary), encoding="utf-8")

    result = {
        "competition_id": competition_name,
        "output_dir": str(output_dir_path),
        "summary_path": str(summary_path),
        "column_report_path": str(column_report_path),
        "notes_path": str(notes_path),
        "task_type": summary["task_type"],
        "num_features": summary["num_features"],
    }

    return result


def main() -> None:
    args = parse_args()
    result = run_analysis(
        competition_name=args.competition_name,
        path=args.path,
        output_dir=args.output_dir,
    )
    logger.info(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
