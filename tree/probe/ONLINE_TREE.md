# Setting A online tree trial

The custom vLLM V0 single-step runner now supports
`TreeSearchParams(branch_trigger_mode="probe", branch_probe_path=".../abcd/A.pt",
branch_probe_threshold=checkpoint["threshold"])`.
Use n=1, no prompt logprobs, one probe checkpoint per batch. This first integration
supports the setting-A hidden-only checkpoint, not B/C/D or a generic classifier.
The model must be the frozen model used to collect the probe features.

The worker uses the same final hidden state and selected-token indices as the
next-token logits. RMS normalization is performed in float32; PCA, scaling and
the linear head are folded into a single vector. Scores are classification
logits, not estimates of U. The gate uses the saved validation threshold and
does not require entropy or WAAD. All legacy trigger defaults are unchanged.

Run on the server from the verl checkout:

```bash
bash tree/scripts/run_probe_online_trial.sh
```

Default output: `/inspire/hdd/global_user/weilongxuan-253108120168/probe_runs/online-A-seed42`.
`WORK_DIR` and `OUTPUT_DIR` can override paths. Question files are atomic and
completed questions are reused only under the same manifest. Restarting a mode
reloads the model. Do not overwrite a checkpoint at a cached path in a live worker.

Protocol: the existing 10 test questions, one seed (42), 16 leaves per question,
2048 response tokens, k=4 sampled without replacement using the existing vLLM
Gumbel-top-k implementation, temperature=1, maximum depth=3, minimum segment=10.
Probe threshold is frozen from A; entropy threshold is the validation 80th
percentile; random trigger probability is 0.2 at each eligible token. Offline
positions were subsampled, so these thresholds need not yield identical online
branch rates. All groups use the same maximum leaf budget. If necessary,
independent full responses top up to exactly 16 answers. Top-ups are labeled
separately. This measures a practical tree-plus-top-up policy.

Each mode writes node segments, reconstructed leaf token sequences, verifier
outputs, truncation, top-ups, generated token counts, and a summary. Generated
tokens count actual sampled node outputs plus top-ups, including replaced
parent tokens; they exclude prefill/re-prefill work. Thus equal leaf and length
caps do not imply equal FLOPs. Wall time is recorded separately.

Primary comparisons: mean correct leaf fraction, number of questions with any
correct answer (oracle coverage, not pass@1), and number with mixed rewards.
Offline U is not re-estimated by this trial. The same 10 test questions have
already informed experimental discussion; results are diagnostic and require
fresh questions before claiming generalization.

Smoke test before the full trial:

```bash
python3 tree/probe/online_tree_trial.py \
  --work-dir "$WORK_DIR" --output-dir "$WORK_DIR/online-smoke" \
  --mode probe --split val --limit 1 --max-response 128 --leaves 4
```

## Fresh 100-question evaluation

`prepare_online_questions.py` streams the original parquet, deduplicates questions,
excludes entire conflicting-answer groups and every question in the probe's
development split, then draws 100 new questions with seed 42. Overlong prompts
are excluded rather than truncated. The selection and data hashes are saved.
The classifier and entropy thresholds still come from the original validation
set; the new questions are used only for evaluation.

```bash
ROOT=/inspire/hdd/global_user/weilongxuan-253108120168
export WORK_DIR="$ROOT/probe_runs/math7b-sample-abcd-seed42"
export OUTPUT_DIR="$ROOT/probe_runs/online-A-fresh100-seed42"
python3 tree/probe/prepare_online_questions.py \
  --work-dir "$WORK_DIR" --output-dir "$OUTPUT_DIR" --num-questions 100
export QUESTIONS_FILE="$OUTPUT_DIR/questions.json"
nohup bash tree/scripts/run_probe_online_trial.sh \
  > "$OUTPUT_DIR/launch.log" 2>&1 < /dev/null &
```

Check `status.txt`, each mode's `.log`, and finally `comparison.json` under the
output directory. Restart with the same exported paths to reuse completed
question artifacts. This is one evaluation seed, not a new probe training run.
