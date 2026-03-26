from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_COMPETITION_ID = "tabular-playground-series-may-2022"
TIME_BUDGET_SECONDS = 300
RANDOM_SEED = 42
RESULTS_PATH = ROOT_DIR / "results.tsv"
RESULTS_HEADER = "commit\tprivate_score\tpublic_score\ttrain_seconds\tmedal\tstatus\tdescription\n"
REQUIRED_PUBLIC_FILES = ("train.csv", "test.csv", "sample_submission.csv")
REQUIRED_PRIVATE_FILES = ("gold_submission.csv",)
SUPPORTED_RAW_PREPARERS = {DEFAULT_COMPETITION_ID}
METADATA_FILENAMES = (
    "checksums.yaml",
    "config.yaml",
    "description.md",
    "description_obfuscated.md",
    "grade.py",
    "leaderboard.csv",
    "prepare.py",
    "utils.py",
)


class InvalidSubmissionError(Exception):
    """Raised when a submission cannot be graded."""


@dataclass(frozen=True)
class CompetitionPaths:
    competition_id: str
    competition_dir: Path
    raw_dir: Path
    public_dir: Path
    private_dir: Path
    train_path: Path
    public_test_path: Path
    sample_submission_path: Path
    gold_submission_path: Path
    private_test_path: Path
    description_path: Path
    leaderboard_path: Path
    config_path: Path
    artifacts_dir: Path
    submissions_dir: Path
    runs_dir: Path

    @property
    def prepared_dir(self) -> Path:
        return self.competition_dir / "prepared"


@dataclass(frozen=True)
class DatasetSummary:
    competition_id: str
    train_rows: int
    public_test_rows: int
    private_rows: int
    num_features: int
    target_mean: float
    description_path: str
    leaderboard_path: str | None


@dataclass(frozen=True)
class ScoreReport:
    score: float
    medal: str
    above_median: bool | None
    estimated_rank: int | None
    total_teams: int | None
    gold_threshold: float | None
    silver_threshold: float | None
    bronze_threshold: float | None
    median_threshold: float | None
    lower_is_better: bool | None


def get_metadata_source_dir(competition_id: str) -> Path | None:
    candidates = [
        ROOT_DIR / competition_id / "prepared",
        ROOT_DIR / "mle-bench" / "mlebench" / "competitions" / competition_id,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def resolve_metadata_path(competition_id: str, filename: str) -> Path:
    local_path = ROOT_DIR / competition_id / "prepared" / filename
    if local_path.exists():
        return local_path
    source_dir = get_metadata_source_dir(competition_id)
    if source_dir is not None:
        return source_dir / filename
    return local_path


def get_competition_paths(competition_id: str = DEFAULT_COMPETITION_ID) -> CompetitionPaths:
    competition_dir = ROOT_DIR / competition_id
    public_dir = competition_dir / "prepared" / "public"
    private_dir = competition_dir / "prepared" / "private"
    artifacts_dir = ROOT_DIR / "artifacts" / competition_id
    submissions_dir = artifacts_dir / "submissions"
    runs_dir = artifacts_dir / "runs"
    return CompetitionPaths(
        competition_id=competition_id,
        competition_dir=competition_dir,
        raw_dir=competition_dir / "raw",
        public_dir=public_dir,
        private_dir=private_dir,
        train_path=public_dir / "train.csv",
        public_test_path=public_dir / "test.csv",
        sample_submission_path=public_dir / "sample_submission.csv",
        gold_submission_path=private_dir / "gold_submission.csv",
        private_test_path=private_dir / "test.csv",
        description_path=resolve_metadata_path(competition_id, "description.md"),
        leaderboard_path=resolve_metadata_path(competition_id, "leaderboard.csv"),
        config_path=resolve_metadata_path(competition_id, "config.yaml"),
        artifacts_dir=artifacts_dir,
        submissions_dir=submissions_dir,
        runs_dir=runs_dir,
    )


def initialize_results_tsv(results_path: Path = RESULTS_PATH) -> Path:
    if not results_path.exists():
        results_path.write_text(RESULTS_HEADER, encoding="utf-8")
    return results_path


def copy_metadata_files(paths: CompetitionPaths) -> None:
    source_dir = get_metadata_source_dir(paths.competition_id)
    if source_dir is None:
        return
    paths.prepared_dir.mkdir(parents=True, exist_ok=True)
    for filename in METADATA_FILENAMES:
        src = source_dir / filename
        dst = paths.prepared_dir / filename
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)


def find_first_file(root: Path, filename: str) -> Path:
    direct = root / filename
    if direct.exists():
        return direct
    matches = sorted(root.rglob(filename))
    if not matches:
        raise FileNotFoundError(f"Could not find `{filename}` under `{root}`.")
    return matches[0]


def extract_zip_to_raw(zip_path: Path, raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(raw_dir)


def prepare_tabular_playground_may_2022(raw_dir: Path, public_dir: Path, private_dir: Path) -> None:
    train_csv = find_first_file(raw_dir, "train.csv")
    old_train = pd.read_csv(train_csv)
    new_train, new_test = train_test_split(old_train, test_size=100_000, random_state=0)

    new_train = new_train.reset_index(drop=True)
    new_test = new_test.reset_index(drop=True)
    new_train["id"] = np.arange(len(new_train))
    new_test["id"] = np.arange(len(new_train), len(new_train) + len(new_test))

    new_test_without_labels = new_test.drop(columns=["target"]).copy()
    gold_submission = new_test[["id", "target"]].copy()
    sample_submission = gold_submission.copy()
    sample_submission["target"] = 0.5

    public_dir.mkdir(parents=True, exist_ok=True)
    private_dir.mkdir(parents=True, exist_ok=True)
    new_train.to_csv(public_dir / "train.csv", index=False)
    new_test_without_labels.to_csv(public_dir / "test.csv", index=False)
    sample_submission.to_csv(public_dir / "sample_submission.csv", index=False)
    new_test.to_csv(private_dir / "test.csv", index=False)
    gold_submission.to_csv(private_dir / "gold_submission.csv", index=False)


def is_prepared(paths: CompetitionPaths) -> bool:
    for filename in REQUIRED_PUBLIC_FILES:
        if not (paths.public_dir / filename).exists():
            return False
    for filename in REQUIRED_PRIVATE_FILES:
        if not (paths.private_dir / filename).exists():
            return False
    return True


def prepare_from_local_sources(paths: CompetitionPaths, force: bool = False) -> None:
    if force and paths.prepared_dir.exists():
        shutil.rmtree(paths.prepared_dir)

    if is_prepared(paths):
        copy_metadata_files(paths)
        return

    zip_candidates = sorted(paths.competition_dir.glob("*.zip"))
    if not paths.raw_dir.exists() or not any(paths.raw_dir.iterdir()):
        if not zip_candidates:
            raise FileNotFoundError(
                f"No prepared dataset found for `{paths.competition_id}` and no local zip file is available. "
                f"Expected either `{paths.prepared_dir}` or a zip inside `{paths.competition_dir}`."
            )
        extract_zip_to_raw(zip_candidates[0], paths.raw_dir)

    if paths.competition_id not in SUPPORTED_RAW_PREPARERS:
        raise NotImplementedError(
            f"Automatic raw->prepared conversion is only implemented for {sorted(SUPPORTED_RAW_PREPARERS)}. "
            f"For `{paths.competition_id}`, place an mle-bench style prepared dataset under `{paths.prepared_dir}` first."
        )

    prepare_tabular_playground_may_2022(paths.raw_dir, paths.public_dir, paths.private_dir)
    copy_metadata_files(paths)


def prepare_for_auroc_metric(
    submission: pd.DataFrame,
    answers: pd.DataFrame,
    id_col: str = "id",
    target_col: str = "target",
) -> dict[str, np.ndarray]:
    if id_col not in answers.columns or target_col not in answers.columns:
        raise InvalidSubmissionError(f"Answers must contain `{id_col}` and `{target_col}` columns.")
    if id_col not in submission.columns or target_col not in submission.columns:
        raise InvalidSubmissionError(f"Submission must contain `{id_col}` and `{target_col}` columns.")
    if len(submission) != len(answers):
        raise InvalidSubmissionError("Submission and answers must have the same number of rows.")

    y_score = pd.to_numeric(submission[target_col], errors="coerce")
    if y_score.isna().any():
        raise InvalidSubmissionError("Submission target column must be numeric.")
    if (y_score < 0).any() or (y_score > 1).any():
        raise InvalidSubmissionError("Submission target column must contain probabilities in [0, 1].")

    submission_sorted = submission.sort_values(id_col).reset_index(drop=True)
    answers_sorted = answers.sort_values(id_col).reset_index(drop=True)
    if not submission_sorted[id_col].equals(answers_sorted[id_col]):
        raise InvalidSubmissionError(f"Submission and answers must contain the same `{id_col}` values.")

    return {
        "y_true": answers_sorted[target_col].to_numpy(),
        "y_score": submission_sorted[target_col].to_numpy(dtype=float),
    }


def grade_submission(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    metric_inputs = prepare_for_auroc_metric(submission, answers)
    return float(roc_auc_score(metric_inputs["y_true"], metric_inputs["y_score"]))


def load_gold_answers(paths: CompetitionPaths) -> pd.DataFrame:
    return pd.read_csv(paths.gold_submission_path)


def read_public_data(paths: CompetitionPaths) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(paths.train_path)
    test_df = pd.read_csv(paths.public_test_path)
    return train_df, test_df


def score_submission(submission: pd.DataFrame, paths: CompetitionPaths) -> ScoreReport:
    score = grade_submission(submission, load_gold_answers(paths))
    return compare_to_leaderboard(score, paths.leaderboard_path)


def compare_to_leaderboard(score: float, leaderboard_path: Path) -> ScoreReport:
    if not leaderboard_path.exists():
        return ScoreReport(
            score=score,
            medal="unavailable",
            above_median=None,
            estimated_rank=None,
            total_teams=None,
            gold_threshold=None,
            silver_threshold=None,
            bronze_threshold=None,
            median_threshold=None,
            lower_is_better=None,
        )

    leaderboard = pd.read_csv(leaderboard_path)
    if "score" not in leaderboard.columns or leaderboard.empty:
        return ScoreReport(
            score=score,
            medal="unavailable",
            above_median=None,
            estimated_rank=None,
            total_teams=None,
            gold_threshold=None,
            silver_threshold=None,
            bronze_threshold=None,
            median_threshold=None,
            lower_is_better=None,
        )

    scores = leaderboard["score"].astype(float).reset_index(drop=True)
    lower_is_better = bool(scores.iloc[0] < scores.iloc[-1])
    num_teams = len(scores)

    def score_at_position(position: int) -> float:
        idx = min(max(position, 1), num_teams) - 1
        return float(scores.iloc[idx])

    if 1 <= num_teams < 100:
        gold_threshold = score_at_position(max(1, int(num_teams * 0.1)))
        silver_threshold = score_at_position(max(1, int(num_teams * 0.2)))
        bronze_threshold = score_at_position(max(1, int(num_teams * 0.4)))
    elif 100 <= num_teams < 250:
        gold_threshold = score_at_position(10)
        silver_threshold = score_at_position(max(1, int(num_teams * 0.2)))
        bronze_threshold = score_at_position(max(1, int(num_teams * 0.4)))
    elif 250 <= num_teams < 1000:
        gold_threshold = score_at_position(10 + int(num_teams * 0.002))
        silver_threshold = score_at_position(50)
        bronze_threshold = score_at_position(100)
    else:
        gold_threshold = score_at_position(10 + int(num_teams * 0.002))
        silver_threshold = score_at_position(max(1, int(num_teams * 0.05)))
        bronze_threshold = score_at_position(max(1, int(num_teams * 0.1)))

    median_threshold = float(scores.median())

    def meets(threshold: float) -> bool:
        return score <= threshold if lower_is_better else score >= threshold

    gold_medal = meets(gold_threshold)
    silver_medal = not gold_medal and meets(silver_threshold)
    bronze_medal = not gold_medal and not silver_medal and meets(bronze_threshold)
    medal = "gold" if gold_medal else "silver" if silver_medal else "bronze" if bronze_medal else "none"
    above_median = score < median_threshold if lower_is_better else score > median_threshold

    if lower_is_better:
        estimated_rank = int((scores < score).sum() + 1)
    else:
        estimated_rank = int((scores > score).sum() + 1)

    return ScoreReport(
        score=score,
        medal=medal,
        above_median=bool(above_median),
        estimated_rank=estimated_rank,
        total_teams=num_teams,
        gold_threshold=gold_threshold,
        silver_threshold=silver_threshold,
        bronze_threshold=bronze_threshold,
        median_threshold=median_threshold,
        lower_is_better=lower_is_better,
    )


def build_dataset_summary(paths: CompetitionPaths) -> DatasetSummary:
    train_df = pd.read_csv(paths.train_path)
    public_test_df = pd.read_csv(paths.public_test_path)
    gold_df = pd.read_csv(paths.gold_submission_path)
    return DatasetSummary(
        competition_id=paths.competition_id,
        train_rows=len(train_df),
        public_test_rows=len(public_test_df),
        private_rows=len(gold_df),
        num_features=len(train_df.columns) - 2,
        target_mean=float(train_df["target"].mean()),
        description_path=str(paths.description_path) if paths.description_path.exists() else "",
        leaderboard_path=str(paths.leaderboard_path) if paths.leaderboard_path.exists() else None,
    )


def validate_prepared_dataset(paths: CompetitionPaths) -> DatasetSummary:
    missing = []
    for filename in REQUIRED_PUBLIC_FILES:
        path = paths.public_dir / filename
        if not path.exists():
            missing.append(str(path))
    for filename in REQUIRED_PRIVATE_FILES:
        path = paths.private_dir / filename
        if not path.exists():
            missing.append(str(path))
    if missing:
        raise FileNotFoundError("Missing required prepared files:\n" + "\n".join(missing))

    train_df = pd.read_csv(paths.train_path)
    public_test_df = pd.read_csv(paths.public_test_path)
    sample_df = pd.read_csv(paths.sample_submission_path)
    gold_df = pd.read_csv(paths.gold_submission_path)

    if "target" not in train_df.columns:
        raise ValueError("Expected `target` column in prepared public train.csv.")
    if "target" in public_test_df.columns:
        raise ValueError("Public test.csv should not contain the target column.")
    if list(sample_df.columns) != ["id", "target"]:
        raise ValueError("sample_submission.csv must have columns [id, target].")
    if list(gold_df.columns) != ["id", "target"]:
        raise ValueError("gold_submission.csv must have columns [id, target].")
    if len(public_test_df) != len(sample_df) or len(sample_df) != len(gold_df):
        raise ValueError("Public test, sample submission, and gold submission must have identical row counts.")
    if not sample_df["id"].equals(gold_df["id"]):
        raise ValueError("sample_submission ids must match gold_submission ids exactly.")

    if paths.private_test_path.exists():
        private_test_df = pd.read_csv(paths.private_test_path)
        if len(private_test_df) != len(gold_df):
            raise ValueError("Private labeled test.csv must align with gold_submission.csv.")
        if "target" not in private_test_df.columns:
            raise ValueError("Private test.csv should retain labels for offline grading.")

    return build_dataset_summary(paths)


def write_context_file(paths: CompetitionPaths, summary: DatasetSummary) -> Path:
    paths.artifacts_dir.mkdir(parents=True, exist_ok=True)
    context = {
        "competition_id": paths.competition_id,
        "time_budget_seconds": TIME_BUDGET_SECONDS,
        "results_path": str(initialize_results_tsv()),
        "summary": asdict(summary),
    }
    if paths.description_path.exists():
        context["description_excerpt"] = paths.description_path.read_text(encoding="utf-8")[:2000]
    context_path = paths.artifacts_dir / "competition_context.json"
    context_path.write_text(json.dumps(context, indent=2), encoding="utf-8")
    return context_path


def ensure_prepared_competition(
    competition_id: str = DEFAULT_COMPETITION_ID,
    force: bool = False,
) -> CompetitionPaths:
    paths = get_competition_paths(competition_id)
    prepare_from_local_sources(paths, force=force)
    summary = validate_prepared_dataset(paths)
    initialize_results_tsv()
    write_context_file(paths, summary)
    paths.submissions_dir.mkdir(parents=True, exist_ok=True)
    paths.runs_dir.mkdir(parents=True, exist_ok=True)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare an mle-bench style competition for AutoML autoresearch.")
    parser.add_argument("--competition-id", default=DEFAULT_COMPETITION_ID, help="Competition id / local folder name.")
    parser.add_argument("--force", action="store_true", help="Rebuild the local prepared directory from raw/zip when possible.")
    args = parser.parse_args()

    paths = ensure_prepared_competition(args.competition_id, force=args.force)
    summary = build_dataset_summary(paths)
    leaderboard = compare_to_leaderboard(summary.target_mean, paths.leaderboard_path)

    print(f"Competition:      {summary.competition_id}")
    print(f"Prepared dir:     {paths.prepared_dir}")
    print(f"Train rows:       {summary.train_rows:,}")
    print(f"Public test rows: {summary.public_test_rows:,}")
    print(f"Private rows:     {summary.private_rows:,}")
    print(f"Num features:     {summary.num_features}")
    print(f"Target mean:      {summary.target_mean:.6f}")
    if paths.description_path.exists():
        print(f"Description:      {paths.description_path}")
    if paths.leaderboard_path.exists():
        print(f"Leaderboard:      {paths.leaderboard_path}")
        print(f"Median score:     {leaderboard.median_threshold:.5f}")
        print(f"Gold threshold:   {leaderboard.gold_threshold:.5f}")
    print(f"Results TSV:      {initialize_results_tsv()}")
    print(f"Time budget:      {TIME_BUDGET_SECONDS}s")
    print()
    print("Ready. Use `python train.py` for experiments.")


if __name__ == "__main__":
    main()
