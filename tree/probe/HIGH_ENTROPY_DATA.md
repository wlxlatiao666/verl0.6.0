# Fresh high-entropy dataset (700 questions)

Source: the existing sample-k experiment's DAPO parquet and frozen
Qwen2.5-Math-7B checkpoint. This constructs data only, without RL or probe fitting.

## Frozen protocol

- Deduplicate normalized question IDs; exclude complete conflicting-answer groups.
- Exclude all 100 old probe-development questions and all 100 online evaluation questions.
- Seed42 shuffle eligible unique questions; assign 500 train, 100 validation,
  100 test **before any rollout or label**. Overlong prompts (>2048) are skipped
  and counted; prompts are never truncated. Question IDs include their grouping.
- Generate two trajectories per question, temperature1, top_p1, max response2048.
- Teacher-force each full trajectory with causal attention and final-norm hidden
  states. Position t uses hidden row prompt_length+t-1, predicting response[t].
  Project vocab logits in batches of32 to bound memory.
- Calibrate entropy P80 using every eligible training token (prefix >=10),
  pooled across training trajectories. Validation/test entropies never enter this
  quantile. Freeze the numerical threshold for all three splits and later online use.
- Sample at most8 positions per trajectory from entropy >= threshold, stratified
  across response progress with randomized selection and spacing >=16 tokens.
  No low-entropy padding, no reward-dependent sampling, no replacement questions
  for low-yield trajectories. This distribution is temporally stratified and is
  not a uniform sample of all high-entropy tokens; comparisons must use the same pool.
- Draw4 distinct next tokens using the same Gumbel-top-k weighted sampling without
  replacement as `branch_sampling=sample`, temperature1.
- Independently continue each candidate4 times for train,8 for validation/test,
  within the original2048-token total response budget. Preserve truncation and
  full outcomes. Compute the existing noise-corrected U and keep zero/all-failed
  positions. Category screening or class weighting belongs to future fitting,
  not validation/test data construction.

Maximum: 1400 trajectories,11200 positions,230400 candidate continuations.
Actual totals depend on high-entropy availability/spacing. Training at4 repeats
has noisier labels than evaluation at8; independent replication can be added later.

## Execution

`build_high_entropy_data.py prepare --work-dir ... --source-dir ...
--exclude-questions .../online-A-fresh100-seed42/questions.json` prepares the split.
Then run `tree/scripts/build_qwen2.5_math7b-high-entropy-data.sh` with WORK_DIR.
The runner acquires a directory lock and queues behind the existing
`run_high_entropy_probe.sh` evaluation, preserving its work and avoiding GPU OOM.
Stages: rollout, full entropy scan, training-only calibration, selected features,
pilot labels (20 train/10 val/10 test), pilot summary, all remaining labels, summary.
Pilot selection is frozen before labels. The runner does not tune selection from
pilot outcomes. It automatically continues the authorized full construction.

Question artifacts are atomically written and provenance-checked for resumption.
A process failure marks status failed; rerunning the same runner resumes completed
questions. Each phase releases its GPU model before the next model loads.

Artifacts under `probe_runs/high-entropy-data700-seed42`:

- config.json, questions.json, prepare_summary.json
- trajectories/, entropy/, entropy_calibration.json
- features/*.json and *.pt (selected hidden states only)
- labels/main/, pilot_summary.json, label_summary.json
- status.txt and stage logs

`counts` contains three disjoint classification categories plus an overlapping
`all_failed` diagnostic; do not sum all four fractions as a probability partition.
These are provisional category rules inherited from A, not ground-truth certainty.

Future RL must exclude all validation/test question IDs and their duplicate rows
from the source training parquet. The source parquet is not modified by this job.

## Verification

Remote unit tests cover seeded high-entropy selection with spacing, sparse/empty
candidate sets without low-entropy padding, and calibration invariance under
arbitrary changes to held-out entropies. Three tests passed. Full GPU stages are
pending behind the existing evaluation at launch time; this is not an end-to-end
completion claim.

## Training continuation (user authorized 2026-09-23)

The previous small online evaluation was stopped at the user's request, retaining
its partial outputs; the new dataset rollout now uses the GPU immediately.
`train_after_high_entropy_data.sh` waits for complete labels, then invokes
`train_high_entropy_data.py`. The trainer reuses the frozen full-trajectory
entropy cutoff (no second P80 filtering), fits RMS/PCA32 and setting-A balanced
L2 classification, calibrates probe P80 on validation, freezes the checkpoint,
and evaluates test U/ranking against entropy and random on the same eligible pool.
It does not start RL. An end-to-end synthetic dataset test passed, including
retention of all eligible rows and exact preservation of the entropy cutoff.
Training state/logs/checkpoint are under `training/` in the dataset directory.
