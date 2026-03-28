# automl-autoresearch

This repo is now a minimal closed loop for **agentic AutoML** on an mle-bench style competition.

The current default competition is `tabular-playground-series-may-2022`.

## Setup

To start a new experiment run, work with the user to:

1. **Agree on a run tag**: use a fresh branch like `automl/<tag>`.
2. **Create the branch**: `git checkout -b automl/<tag>` from the current main branch.
3. **Read the in-scope files**:
   - `README.md` - repository context.
   - `prepare.py` - fixed infrastructure: dataset discovery/prep, grading helpers, leaderboard comparison.
   - `train.py` - the only ML file you modify.
   - `analysis.ipynb` - experiment analysis and progress plot.
   - `tabular-playground-series-may-2022/prepared/description.md` - task statement.
4. **Prepare the competition assets**: run `uv run python prepare.py`.
   - This should verify that `./tabular-playground-series-may-2022/prepared/` is usable.
   - If only a local zip/raw dataset exists, `prepare.py` will materialize the mle-bench style `public/` and `private/` splits when possible.
5. **Initialize `results.tsv`**: `prepare.py` creates it if needed.
6. **Confirm and go**: after the dataset is ready, Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a CPU-only machine. You launch it simply as: `uv run train.py`.

What you CAN do:

- Modify `train.py` — this is the only file you edit. Everything is fair game: data cleaning, feature engineering, model architecture, hyperparameters, ensembling, etc.

What you CANNOT do:

- Modify `prepare.py`. It is read-only. It contains the fixed evaluation, data loading, tokenizer, and training constants.
- Install new packages or add dependencies. You can only use what's already in pyproject.toml.
- Modify the evaluation harness. The `grade_submission` function in `prepare.py` is the ground truth metric.

The job of `train.py` is:

- load the mle-bench style competition data
- run a local AutoML search / model selection procedure
- generate a valid `submission.csv`
- report both:
  - `public_score`: the local public proxy score (OOF / CV on public train)
  - `private_score`: the offline hidden score against `private/gold_submission.csv`
- compare the private score to the historical leaderboard and print the medal bucket

**The goal is simple: get the highest `private_score` **, and for this competition **higher is better** because the metric is ROC AUC.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude.

## First Run

The first run: Your very first run should always be to establish the baseline, so you will run the training script as is.

```bash
uv run python train.py > run.log 2>&1
```

Then inspect the summary:

```bash
rg "^(public_score|private_score|medal|train_seconds):" run.log
```

## Output Format

A successful run prints a summary block like:

```text
---
competition_id:    tabular-playground-series-may-2022
feature_version:   v1
public_score:      0.987654
private_score:     0.988765
medal:             silver
estimated_rank:    123
train_seconds:     92.4
selected_models:   hgb_wide, logreg_l2
submission_path:   ...
run_summary_path:  ...
```

If `private_score` is missing, assume the run crashed or produced an invalid submission.

## Logging Results

Log every experiment to `results.tsv` as tab-separated values with this header:

```text
commit	private_score	public_score	train_seconds	medal	status	description
```

Field meanings:

1. git commit hash (short, 7 chars)
2. private score achieved, use `0.000000` for crashes
3. public proxy score achieved, use `0.000000` for crashes
4. training seconds, use `0.0` for crashes
5. medal bucket: `gold`, `silver`, `bronze`, `none`, or `unavailable`
6. status: `keep`, `discard`, or `crash`
7. short experiment description

Example:

```text
commit	private_score	public_score	train_seconds	medal	status	description
abc1234	0.987650	0.986900	94.2	silver	keep	baseline blend of two HGB models and logistic regression
bcd2345	0.988110	0.987050	101.7	silver	keep	add extra string interaction features
cde3456	0.987100	0.986500	88.4	none	discard	reduce search rows for faster iterations
```

Do not commit `results.tsv`.

## Experiment Loop

Loop autonomously after setup:

1. Check the current branch and commit.
2. Modify `train.py` with one clear AutoML idea.
3. Commit the change.
4. Run the experiment:

```bash
uv run python train.py > run.log 2>&1
```

5. Read out the results:

```bash
rg "^(public_score|private_score|medal|train_seconds):" run.log
```

6. If the summary block is missing, inspect the failure:

```bash
tail -n 80 run.log
```

7. Append the result to `results.tsv`.
8. If `private_score` improved, keep the commit and continue from there.
9. If `private_score` is equal or worse, reset back to the previous best commit.

## Research Guidance

Good `train.py` ideas include:

- improving tabular feature engineering
- changing candidate model families or hyperparameters
- altering search budget, folds, or blend logic
- smarter ensembling and calibration
- simplifying the pipeline while keeping or improving `private_score`

Keep the code readable. Small, clean improvements beat large messy changes with no clear gain.

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

*Timeout*: Each experiment should take ~5 minutes total (+ a few seconds for startup and eval overhead). If a run exceeds 10 minutes, kill it and treat it as a failure (discard and revert).

*Crashes*: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

*NEVER STOP*: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working indefinitely until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~5 minutes then you can run approx 12/hour, for a total of about 100 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!
