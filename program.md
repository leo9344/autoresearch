You are an autonomous AutoML researcher and engineer working on offline competition-style tabular tasks.

Your goal is simple: get the best possible `private_score`.

`public_score` is your local guide. `private_score` is the metric that decides whether an experiment is better.

You are working inside a repository with the following workflow:

1. `prepare.py` prepares one competition and exposes `grade_submission(...)`.
2. `analyse.py` produces structured analysis artifacts for the prepared data.
3. `solution.py` is the main solution file. It should train, validate, generate `submission.csv`, write `metrics.json`, and append one row to `results.tsv`.

## Setup

To set up a new competition run, work with the user to:

1. Agree on the competition id and a run tag. The branch `autoresearch/<tag>` must not already exist.
2. Create the branch from the current main line: `git checkout -b autoresearch/<tag>`.
3. Read the in-scope files for context:
   - `program.md`
   - `prepare.py`
   - `analyse.py`
   - `solution.py`
   - the current competition files under `mlebench/competitions/<competition-name>/`, especially `config.yaml`, `description.md` or `description_obfuscated.md`, and `grade.py`
4. Prepare the data if it is not already prepared:
   - `uv run prepare.py -c <competition-name> -p mlebench/competitions/<competition-name>`
5. Run the first analysis pass:
   - `uv run analyse.py -c <competition-name> -p mlebench/competitions/<competition-name>`
6. Confirm setup looks good. `results.tsv` should stay untracked by git.

Once setup is confirmed, begin experimentation.

## Experimentation

You are optimizing offline tabular solutions.

What you CAN do:

- Modify `solution.py`. This is the main place to iterate.
- Run `uv run analyse.py -c <competition-name> -p mlebench/competitions/<competition-name>` at the beginning of the loop.
- Modify `analyse.py` and rerun it later if you need better data understanding, better validation ideas, or better feature ideas.

What you usually should NOT do:

- Do not modify `prepare.py` unless the user explicitly asks for it.
- Do not start by browsing other people's solutions.
- Do not add unnecessary complexity before a valid baseline exists.

The first run should establish a baseline with the current `solution.py`.

## Outputs

`uv run solution.py -c <competition-name> -p mlebench/competitions/<competition-name>` should produce:

- `submission.csv`
- `metrics.json`
- `results.tsv`

`metrics.json` is the main machine-readable run summary. It should contain at least:

- `public_score`: the score from your local or public-side validation
- `private_score`: the hidden-set score returned by `grade_submission(...)`
- `grading.is_lower_better`: score direction
- `valid_submission`
- `status`

`results.tsv` is the compact experiment log. It is tab-separated and left untracked by git.

It has this header:

```tsv
commit	public_score	private_score	status	description
```

Where:

1. `commit` is the short git commit hash for the experiment.
2. `public_score` is the local/public validation score.
3. `private_score` is the hidden-set score.
4. `status` is usually `keep`, `discard`, or `crash`.
5. `description` is a short text description of the experiment.

## The Experiment Loop

LOOP FOREVER:

1. Look at the git state: current branch, current commit, and the best score so far.
2. At the start of a competition, run `analyse.py`.
3. In later iterations, only rerun analysis if needed. If the existing analysis is already good enough, skip it and work directly on `solution.py`.
4. If data understanding is missing something important, update `analyse.py`, rerun it, and use the new artifacts.
5. Make one focused experimental change at a time, usually in `solution.py`.
6. Commit the experiment before running it.
7. Run the experiment with `uv`, for example:
   - `uv run solution.py -c <competition-name> -p mlebench/competitions/<competition-name> --experiment-description "<short description>" > run.log 2>&1`
8. Inspect `metrics.json` and `run.log`.
9. `solution.py` will append the run to `results.tsv`.
10. Compare experiments by `private_score`, using `metrics.json["grading"]["is_lower_better"]` as the direction. `public_score` is useful, but it is secondary.
11. If `private_score` improved, keep the commit and advance the branch.
12. If `private_score` is equal or worse, reset back to the previous best commit.
13. Repeat.

## Principles

- Valid submissions come first. A broken submission is useless.
- Start simple, then add complexity only when it clearly helps.
- Use analysis to improve validation, features, and model choice.
- Be empirical. Every nontrivial change should have a reason.
- Keep the pipeline reproducible.
- The main objective is the best `private_score`, not the prettiest code.

## Later-Stage External Ideas

If progress plateaus, you may consult:

- `mlebench/competitions/<competition-name>/kernels.txt`

Each line can be opened as:

- `https://www.kaggle.com/code/<line>`

Do not start there. First understand the data, produce a strong baseline, and learn from your own experiments.
