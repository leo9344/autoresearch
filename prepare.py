from __future__ import annotations

import argparse
import json
import shutil
import webbrowser
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from mlebench.grade_helpers import Grader, InvalidSubmissionError
from mlebench.utils import (
    authenticate_kaggle_api,
    extract,
    get_logger,
    import_fn,
    is_empty,
    load_answers,
    load_yaml,
    read_csv,
)

logger = get_logger(__name__)

REPO_ROOT = Path(__file__).resolve().parent
COMPETITIONS_DIR = REPO_ROOT / "mlebench" / "competitions"


@dataclass(frozen=True)
class CompetitionContext:
    competition_id: str
    source_dir: Path
    output_root: Path
    raw_dir: Path
    public_dir: Path
    private_dir: Path
    description_source: Path
    prepare_fn: Callable[[Path, Path, Path], object]
    grader: Grader
    answers_path: Path
    sample_submission_path: Path
    gold_submission_path: Path
    leaderboard_path: Path
    expected_output_files: dict[str, Path]


@dataclass(frozen=True)
class GradeResult:
    competition_id: str
    metric_name: str
    score: float | None
    submission_path: str
    submission_exists: bool
    valid_submission: bool
    error_message: str | None
    is_lower_better: bool | None
    gold_threshold: float | None
    silver_threshold: float | None
    bronze_threshold: float | None
    median_threshold: float | None
    any_medal: bool
    gold_medal: bool
    silver_medal: bool
    bronze_medal: bool
    above_median: bool
    answers_path: str
    leaderboard_path: str | None
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "competition_id": self.competition_id,
            "metric_name": self.metric_name,
            "score": self.score,
            "submission_path": self.submission_path,
            "submission_exists": self.submission_exists,
            "valid_submission": self.valid_submission,
            "error_message": self.error_message,
            "is_lower_better": self.is_lower_better,
            "gold_threshold": self.gold_threshold,
            "silver_threshold": self.silver_threshold,
            "bronze_threshold": self.bronze_threshold,
            "median_threshold": self.median_threshold,
            "any_medal": self.any_medal,
            "gold_medal": self.gold_medal,
            "silver_medal": self.silver_medal,
            "bronze_medal": self.bronze_medal,
            "above_median": self.above_median,
            "answers_path": self.answers_path,
            "leaderboard_path": self.leaderboard_path,
            "created_at": self.created_at,
        }


def list_competition_ids() -> list[str]:
    return sorted(config.parent.name for config in COMPETITIONS_DIR.glob("*/config.yaml"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and prepare a single MLE-bench competition dataset."
    )
    parser.add_argument(
        "-c",
        "--competition-name",
        required=True,
        help="Competition ID, for example `tabular-playground-series-may-2022`.",
    )
    parser.add_argument(
        "-p",
        "--path",
        type=Path,
        default=None,
        help=(
            "Output root directory. Defaults to "
            "`./mlebench/competitions/<competition-name>`."
        ),
    )
    parser.add_argument(
        "--zip-file",
        type=Path,
        default=None,
        help="Use an existing Kaggle zip instead of downloading it.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Do not hit Kaggle. Requires an existing zip file or a populated `raw/` directory.",
    )
    parser.add_argument(
        "--keep-raw",
        action="store_true",
        help="Keep the extracted `raw/` directory after preparation finishes.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild `prepared/` even if it already exists.",
    )
    parser.add_argument(
        "--submission",
        type=Path,
        default=None,
        help="Optional submission CSV to grade after the dataset is ready.",
    )
    return parser.parse_args()


def strip_competition_prefix(competition_id: str, relative_path: str) -> Path:
    path = Path(relative_path)
    if path.parts and path.parts[0] == competition_id:
        return Path(*path.parts[1:])
    return path


def resolve_output_root(competition_id: str, output_root: str | Path | None) -> Path:
    if output_root is None:
        return (COMPETITIONS_DIR / competition_id).resolve()
    return Path(output_root).expanduser().resolve()


def resolve_metadata_path(output_root: Path, source_dir: Path, filename: str) -> Path:
    destination_path = (output_root / filename).resolve()
    if destination_path.exists() or output_root == source_dir:
        return destination_path
    return (source_dir / filename).resolve()


def build_context(competition_id: str, output_root: Path) -> CompetitionContext:
    output_root = output_root.resolve()
    source_dir = COMPETITIONS_DIR / competition_id
    config_path = source_dir / "config.yaml"

    if not config_path.is_file():
        valid_ids = ", ".join(list_competition_ids())
        raise FileNotFoundError(
            f"Unknown competition `{competition_id}`. Valid options: {valid_ids}"
        )

    config = load_yaml(config_path)
    prepare_fn = import_fn(config["preparer"])
    grader = Grader.from_dict(config["grader"])
    description_source = REPO_ROOT / config["description"]
    dataset_paths = {
        key: output_root / strip_competition_prefix(competition_id, relative_path)
        for key, relative_path in config["dataset"].items()
    }
    expected_output_files = dict(dataset_paths)

    return CompetitionContext(
        competition_id=competition_id,
        source_dir=source_dir,
        output_root=output_root,
        raw_dir=(output_root / "raw").resolve(),
        public_dir=(output_root / "prepared" / "public").resolve(),
        private_dir=(output_root / "prepared" / "private").resolve(),
        description_source=description_source.resolve(),
        prepare_fn=prepare_fn,
        grader=grader,
        answers_path=dataset_paths["answers"].resolve(),
        sample_submission_path=dataset_paths["sample_submission"].resolve(),
        gold_submission_path=dataset_paths.get("gold_submission", dataset_paths["answers"]).resolve(),
        leaderboard_path=resolve_metadata_path(output_root, source_dir, "leaderboard.csv"),
        expected_output_files=expected_output_files,
    )


def ensure_dirs(context: CompetitionContext) -> None:
    context.output_root.mkdir(parents=True, exist_ok=True)
    context.raw_dir.mkdir(parents=True, exist_ok=True)
    context.public_dir.mkdir(parents=True, exist_ok=True)
    context.private_dir.mkdir(parents=True, exist_ok=True)


def sync_competition_metadata(context: CompetitionContext, force: bool) -> None:
    if context.source_dir == context.output_root:
        return

    for source_path in context.source_dir.iterdir():
        if not source_path.is_file():
            continue

        destination_path = context.output_root / source_path.name
        if destination_path.exists() and not force:
            continue

        shutil.copy2(source_path, destination_path)


def prepared_dataset_exists(context: CompetitionContext) -> bool:
    if not context.public_dir.is_dir() or is_empty(context.public_dir):
        return False
    if not context.private_dir.is_dir() or is_empty(context.private_dir):
        return False
    return all(path.is_file() for path in context.expected_output_files.values())


def resolve_zip_path(
    competition_id: str,
    output_root: Path,
    explicit_zip_file: Path | None,
) -> Path | None:
    if explicit_zip_file is not None:
        zip_path = explicit_zip_file.expanduser().resolve()
        if not zip_path.is_file():
            raise FileNotFoundError(f"Zip file not found: `{zip_path}`")
        return zip_path

    preferred = (output_root / f"{competition_id}.zip").resolve()
    if preferred.is_file():
        return preferred

    zip_files = sorted(path.resolve() for path in output_root.glob("*.zip") if path.is_file())
    if len(zip_files) == 1:
        return zip_files[0]
    if len(zip_files) > 1:
        raise ValueError(
            f"Found multiple zip files under `{output_root}`. Pass `--zip-file` explicitly."
        )
    return None


def need_to_accept_rules(error_message: str) -> bool:
    return "You must accept this competition" in error_message


def prompt_user_to_accept_rules(competition_id: str) -> None:
    response = input("Open the Kaggle competition rules page now? [y/N]: ").strip().lower()
    if response != "y":
        raise RuntimeError(
            "Kaggle competition rules must be accepted before the dataset can be downloaded."
        )

    webbrowser.open(f"https://www.kaggle.com/c/{competition_id}/rules")
    input("Press Enter after you have accepted the rules...")


def download_competition_zip(context: CompetitionContext) -> Path:
    logger.info(
        "Downloading `%s` into `%s`...",
        context.competition_id,
        context.output_root,
    )

    api = authenticate_kaggle_api()

    # Import lazily so Kaggle credentials are not requested at module import time.
    from kaggle.rest import ApiException

    try:
        api.competition_download_files(
            competition=context.competition_id,
            path=context.output_root,
            quiet=False,
            force=False,
        )
    except ApiException as exc:
        if not need_to_accept_rules(str(exc)):
            raise
        logger.warning("Competition rules must be accepted before downloading.")
        prompt_user_to_accept_rules(context.competition_id)
        return download_competition_zip(context)

    zip_path = resolve_zip_path(context.competition_id, context.output_root, explicit_zip_file=None)
    if zip_path is None:
        raise FileNotFoundError(
            f"Expected a downloaded zip under `{context.output_root}`, but none was found."
        )
    return zip_path


def ensure_raw_data(
    context: CompetitionContext,
    explicit_zip_file: Path | None,
    skip_download: bool,
) -> Path | None:
    if context.raw_dir.is_dir() and not is_empty(context.raw_dir):
        logger.info("Reusing existing raw data in `%s`.", context.raw_dir)
        return None

    zip_path = resolve_zip_path(
        context.competition_id,
        context.output_root,
        explicit_zip_file=explicit_zip_file,
    )
    if zip_path is None and skip_download:
        raise FileNotFoundError(
            "No raw data or zip file is available, and `--skip-download` was set."
        )
    if zip_path is None:
        zip_path = download_competition_zip(context)

    logger.info("Extracting `%s` to `%s`...", zip_path, context.raw_dir)
    extract(zip_path, context.raw_dir, recursive=False)
    return zip_path


def copy_description(context: CompetitionContext) -> None:
    description_output = context.public_dir / "description.md"
    description_output.write_text(context.description_source.read_text(encoding="utf-8"), encoding="utf-8")


def validate_grading_ready(context: CompetitionContext) -> None:
    if not context.answers_path.is_file():
        raise FileNotFoundError(
            f"Answers file not found for `{context.competition_id}`: `{context.answers_path}`"
        )


def get_rank_info(context: CompetitionContext, score: float | None) -> dict[str, Any]:
    if not context.leaderboard_path.is_file():
        return {
            "gold_medal": False,
            "silver_medal": False,
            "bronze_medal": False,
            "above_median": False,
            "gold_threshold": None,
            "silver_threshold": None,
            "bronze_threshold": None,
            "median_threshold": None,
            "is_lower_better": None,
        }

    leaderboard = read_csv(context.leaderboard_path)
    rank_info = context.grader.rank_score(score, leaderboard)
    return {
        **rank_info,
        "is_lower_better": context.grader.is_lower_better(leaderboard),
    }


def validate_outputs(context: CompetitionContext) -> None:
    if not context.public_dir.is_dir() or is_empty(context.public_dir):
        raise RuntimeError(f"Prepared public directory is missing or empty: `{context.public_dir}`")
    if not context.private_dir.is_dir() or is_empty(context.private_dir):
        raise RuntimeError(
            f"Prepared private directory is missing or empty: `{context.private_dir}`"
        )

    missing = {
        key: path
        for key, path in context.expected_output_files.items()
        if not path.is_file()
    }
    if missing:
        expected = ", ".join(f"{key} -> {path}" for key, path in missing.items())
        raise RuntimeError(f"Preparation finished but required files are missing: {expected}")


def prepare_competition(
    context: CompetitionContext,
    explicit_zip_file: Path | None,
    skip_download: bool,
    keep_raw: bool,
    force: bool,
) -> None:
    ensure_dirs(context)
    sync_competition_metadata(context, force=force)

    if prepared_dataset_exists(context) and not force:
        logger.info(
            "Prepared dataset already exists at `%s`. Use `--force` to rebuild it.",
            context.output_root / "prepared",
        )
        return

    if force and (context.output_root / "prepared").exists():
        logger.info("Removing existing prepared data under `%s`...", context.output_root / "prepared")
        shutil.rmtree(context.output_root / "prepared")
        ensure_dirs(context)

    ensure_raw_data(
        context=context,
        explicit_zip_file=explicit_zip_file,
        skip_download=skip_download,
    )

    logger.info(
        "Running `%s` from `%s`...",
        context.prepare_fn.__name__,
        context.prepare_fn.__module__,
    )
    context.prepare_fn(
        raw=context.raw_dir,
        public=context.public_dir,
        private=context.private_dir,
    )

    copy_description(context)
    validate_outputs(context)

    if not keep_raw and context.raw_dir.exists():
        logger.info("Removing raw data under `%s`...", context.raw_dir)
        shutil.rmtree(context.raw_dir)


def prepare_dataset(
    competition_name: str,
    path: str | Path | None = None,
    zip_file: str | Path | None = None,
    skip_download: bool = False,
    keep_raw: bool = False,
    force: bool = False,
) -> CompetitionContext:
    """Prepare one competition dataset and return its resolved local context."""

    explicit_zip_file = None
    if zip_file is not None:
        explicit_zip_file = Path(zip_file).expanduser().resolve()

    context = build_context(competition_name, resolve_output_root(competition_name, path))
    prepare_competition(
        context=context,
        explicit_zip_file=explicit_zip_file,
        skip_download=skip_download,
        keep_raw=keep_raw,
        force=force,
    )
    return context


def grade_submission_with_context(
    submission_path: str | Path,
    context: CompetitionContext,
) -> GradeResult:
    validate_grading_ready(context)

    submission_path = Path(submission_path).expanduser().resolve()
    submission_exists = submission_path.is_file() and submission_path.suffix.lower() == ".csv"

    score = None
    valid_submission = False
    error_message = None

    if submission_exists:
        try:
            submission_df = read_csv(submission_path)
            answers = load_answers(context.answers_path)
            score = round(float(context.grader.grade_fn(submission_df, answers)), 5)
            valid_submission = True
        except InvalidSubmissionError as exc:
            error_message = str(exc)
            logger.warning("Invalid submission for `%s`: %s", context.competition_id, exc)
        except Exception as exc:
            error_message = f"Unexpected error during grading: {exc}"
            logger.exception("Unexpected grading error for `%s`.", context.competition_id)
    else:
        error_message = f"Submission file not found or not a CSV: `{submission_path}`"

    rank_info = get_rank_info(context, score)

    return GradeResult(
        competition_id=context.competition_id,
        metric_name=context.grader.name,
        score=score,
        submission_path=str(submission_path),
        submission_exists=submission_exists,
        valid_submission=valid_submission,
        error_message=error_message,
        is_lower_better=rank_info["is_lower_better"],
        gold_threshold=rank_info["gold_threshold"],
        silver_threshold=rank_info["silver_threshold"],
        bronze_threshold=rank_info["bronze_threshold"],
        median_threshold=rank_info["median_threshold"],
        any_medal=bool(
            rank_info["gold_medal"] or rank_info["silver_medal"] or rank_info["bronze_medal"]
        ),
        gold_medal=bool(rank_info["gold_medal"]),
        silver_medal=bool(rank_info["silver_medal"]),
        bronze_medal=bool(rank_info["bronze_medal"]),
        above_median=bool(rank_info["above_median"]),
        answers_path=str(context.answers_path),
        leaderboard_path=str(context.leaderboard_path) if context.leaderboard_path.is_file() else None,
        created_at=datetime.now().isoformat(),
    )


def grade_submission(
    submission_path: str | Path,
    competition_name: str,
    path: str | Path | None = None,
) -> GradeResult:
    """Grade a submission CSV against the prepared hidden labels for one competition."""

    context = build_context(competition_name, resolve_output_root(competition_name, path))
    return grade_submission_with_context(submission_path, context)


def main() -> None:
    args = parse_args()

    context = prepare_dataset(
        competition_name=args.competition_name,
        path=args.path,
        zip_file=args.zip_file,
        skip_download=args.skip_download,
        keep_raw=args.keep_raw,
        force=args.force,
    )

    logger.info("Preparation finished.")
    logger.info("Output root: %s", context.output_root)
    logger.info("Public split: %s", context.public_dir)
    logger.info("Private split: %s", context.private_dir)

    if args.submission is not None:
        grade_result = grade_submission_with_context(args.submission, context)
        logger.info("Grading finished.")
        logger.info(json.dumps(grade_result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
