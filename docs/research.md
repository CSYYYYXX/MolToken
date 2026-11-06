# Research and rebuttal code

This repository contains the MolToken implementation and its research drivers.
The graph model, frozen production tokenizer and experimental ablation tokenizers
are distinct interfaces. Never use an ablation vocabulary to decode a production
checkpoint, or substitute a production tokenizer for an E2/E3 checkpoint.

## Source map

| Area | Entry points |
| --- | --- |
| Production tokenizer, model loading and decoding | `moltoken/`, `pipeline/`, `src/` |
| Corpus preparation | `cli/data/` |
| Pretraining and held-out loss | `cli/pretrain/`, `experiments/pretraining/` |
| Production two-stage fine-tuning | `cli/sft/` |
| Generation and evaluator adapters | `cli/evaluate/` |
| Scaling matrices, accounting, fits and analysis | `experiments/scaling/` |
| Representation diagnostics | `experiments/representation/` |
| E2a ordering and rotation ablations | `experiments/rebuttal/common.py`, `serialize_tokenizers_worker.py`, `fit_bpe_worker.py`, `rotation_ablation_worker.py`, `rotation_full_cohort_worker.py` |
| E2b/E3 graph/native representation and pretraining controls | `experiments/rebuttal/prepare_cohorts_worker.py`, `prepare_training_worker.py`, `model.py`, `train_worker.py`, `run_gpu_suite.py`, `evaluate_worker.py` |
| GB1 / promoter / MRL transfer | `experiments/rebuttal/transfer_prepare_worker.py`, `transfer_model.py`, `transfer_train_worker.py`, `transfer_suite_worker.py` |
| Saved-result validation and diagnostics | `experiments/rebuttal/transfer_audit_worker.py`, `transfer_isolation_probe_worker.py`, `validate_and_report_worker.py`, `finalize_results_worker.py` |

These drivers include historical research utilities; their presence is not
permission to rerun a campaign or a claim that all planned experiments were
completed. E1 independent-test reevaluation and E4 held-out-budget extrapolation
were not recorded as completed in the source campaign. The transfer campaign
recorded 32 of 36 completed runs; four stopped at the packed/single feature
check. This release preserves that check and does not relabel failures as passes.
Do not silently relax the check or change the comparison protocol for a rerun.

The production two-stage SFT protocol and the E2/E3 direct Full fine-tuning
control are different experiments. Keep their settings and result labels distinct.

## Environment and data contract

Install the package with `python -m pip install -e '.[train,data,test,bio-eval]'`.
Rebuttal GPU training additionally needs an xFormers build compatible with the
selected PyTorch/CUDA stack. Official metrics need the separately documented
[evaluator dependencies](benchmarks.md) and external model assets. No data,
checkpoint tensors, private migration catalog or cluster credentials are included.

Run Python drivers from the repository root, or install this checkout editable.
Use each driver's `--help` to specify input, run and output paths. The rebuttal
scripts take a `--run` directory containing the campaign's cohorts, tokenizer
artifacts, encoded streams, model states and manifests; transferring weights
alone does not reconstruct those inputs.

Shell wrappers accept `MOL` (absolute local asset directory), `RUN`, and
`PYTHON` (interpreter). `WORKERS` controls CPU preparation parallelism. For example:

```bash
export MOL=/absolute/path/to/molscaling
export RUN="$MOL/rebuttal_experiments/my_run"
export PYTHON="$(command -v python)"
bash experiments/rebuttal/build_runtime_worker.sh
python -m experiments.rebuttal.prepare_cohorts_worker --help
python -m experiments.rebuttal.train_worker --help
```

`build_runtime_worker.sh` builds and tests the separate ordering-ablation
extension and integer-BPE executable; it does not launch a training campaign.
A C++17 compiler and matching Python development headers are required. For a
relocated Python runtime, set `PYTHON_INCLUDE_DIR` to its development-header directory.
The package install also builds `_rebuttal_fddfs_cpp`. Production tokenization
continues to use `pipeline.serialization._stackk_cpp`.

Set `MOLTOKEN_REQUIRE_WORKER=1` only on clusters where compute must run on hosts
whose names contain `worker`; standalone machines do not need that hostname rule.
Historical scheduler launchers under `experiments/scaling/` require the original
scheduler commands and explicitly configured mounts/remote host. They are not
portable launch defaults. Use the Python matrix/training/analysis drivers on a
new server and supply local paths. Any `./storage` path is a local example layout,
not a downloadable asset location. Scaling paths can be overridden through
`HOLOMOL_WORKSPACE_ROOT`, `HOLOMOL_SCALING_DATA_ROOT`, and
`HOLOMOL_SCALING_RUN_ROOT`.

## Validation commands

```bash
python -m pytest -q
python tests/smoke_training_pipeline.py --work-dir /absolute/new/smoke-directory
```

The second command uses tiny synthetic models and inputs, executes corpus
encoding/indexing, pretraining, loss evaluation, SFT materialization, Alignment,
Full and sampling, and writes a per-step report. It creates a new directory and
refuses to overwrite an existing one. It validates execution, not benchmark
quality. GPU packed attention and full benchmark reproduction require their
actual hardware and data and are not established by CPU smoke tests.

Pin the Git commit in every experiment record together with the data/tokenizer
manifest, checkpoint lineage, environment versions, launch arguments and seeds.
Uncommitted edits should be captured as a patch, not described as the unchanged
published version.
