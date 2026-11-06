# Pipeline and data interfaces

Run commands from the repository root. Paths in the examples are relative to
that directory. Download the required data and encoder weights before training.

## 1. Graphs and the frozen tokenizer

`MolTokenizer` uses the bundled frequency guide, base vocabulary and merge table.
SMILES is parsed by RDKit. Peptide/protein residues and DNA/RNA nucleotides expand
directly from the templates in `pipeline/graph/chem_graph_builder.py`; they do
not pass through a FASTA-to-SMILES conversion. Peptide bonds and phosphodiester
bonds join the residue graphs. FDDFS first constructs the guided spanning
forest and ring labels, then serializes the forest. Integer BPE compresses that
sequence. Inversion expands BPE before parsing FDDFS.

```python
from moltoken import MolTokenizer
t = MolTokenizer()
ids = t.encode("AGK", "protein")
assert t.decode_sequence(ids, "protein") == "AGK"
graph = t.decode_graph(ids)
```

The graph holds heavy-atom features and bonds, not 3D coordinates. The standard
biopolymer decoder accepts the canonical linear residue grammar, not arbitrary
modified or branched polymers. Invalid sequences/graphs raise an error; the
tokenizer does not silently substitute residues.

`encode()` returns raw BPE IDs. `model_ids=True` adds the model offset and
BOS/EOS. For the checkpoint's vocabulary convention, see
[checkpoints.md](checkpoints.md).

## 2. Corpus preparation

For a new, already selected corpus, use JSONL with one record per line:

```json
{"id":"example-1","domain":"protein","sequence":"ACDEFG"}
```

The five domains are `small_molecule`, `peptide`, `protein`, `dna`, `rna`.
For small molecules `sequence` contains SMILES (`smiles` is also accepted).
Prepare training and held-out data as separate input files.

```bash
moltoken encode-corpus --input examples/toy_corpus.jsonl \
  --output data/encoded/shard_000.bin
python -m cli.data.build_compact_int_index \
  --input-shards data/encoded/shard_000.bin --index-dir data/index
```

The output uses the `YTTMIBPE` compact-int format:
little-endian header `<8sIQQi` (magic, version, record count, token count, maximum
raw ID), followed by `<I length>` and `length` int32 IDs per record. A JSONL
sidecar preserves IDs and domains. Input records should already be normalized,
deduplicated and assigned to a split.

The large-corpus ingestion route is also included:

| Step | Module / configuration |
|---|---|
| Download source files | `cli.data.download_sources`, `configs/data/current/sources.yaml` |
| Normalize to domain JSONL | `cli.data.process_all` / `process_sharded`, `configs/data/current/process.yaml` |
| Validate and deduplicate | `cli.data.clean_dedup_audit` / `clean_dedup_sharded` |
| Shard the cleaned corpus | `cli.data.shard_clean_normv2_all` |
| Expand heavy-atom graphs | `cli.data.build_heavy_atom_graphs`, graph YAML in `configs/data/current/` |
| Count guide and serialize FDDFS | `cli.pretrain.serialize_fddfs_full` |
| Apply or train integer BPE | `tools/yttm_int_bpe/`; see [tokenizer_training.md](tokenizer_training.md) |
| Build random-access/domain indexes | `cli.data.build_compact_int_index`, `build_domain_id_sidecars_from_manifest_assignment` |

Each module exposes its arguments through `python -m <module> --help`. Select
sources and set `root` in the corresponding YAML configuration. The source
registry records download locations and dataset license information. Install
the `zstd` command-line executable for `.zst` streaming.

The default data location is `./data`; `MOLTOKEN_DATA_ROOT` overrides it.
`moltoken assets --data-root ./data` installs the bundled tokenizer in the data
layout used by the task materializers. Encoder locations are supplied separately.

## 3. Pretraining, scaling runs and held-out loss

`examples/pretrain.sh` is a configurable single-/multi-GPU launch example.
The full implementation is `cli.pretrain.train_qwen3_pretrain`. It includes
the Qwen3 configuration, packed xFormers attention, online padding, streaming
compact data, domain logging, distributed training, schedules and checkpointing.

```bash
DATA=./data/encoded INDEX=./data/index OUT=./runs/pretrain \
  GPUS=1 BATCH=8 ACCUM=4 STEPS=1000 bash examples/pretrain.sh
```

Set batch size, gradient accumulation and training steps for the chosen corpus
and compute budget. `packed-xformers` isolates attention between samples. The
`hf` implementation also supports `online-pad`, with one sample per row.

Single-domain scaling uses the same trainer with that domain's selected files
or domain manifest and the desired model dimensions/budget. Multi-domain runs
use the combined corpus. `--init-checkpoint` initializes model weights without
restoring optimizer state.

```bash
moltoken evaluate-loss --checkpoint ./runs/pretrain/final \
  --token-files './data/heldout/*.bin' --index-dir ./data/heldout_index \
  --output ./results/heldout_loss.json --device cuda --dtype bf16 --batch-size 8
```

This command uses the original sample construction and token-weighted NLL.
By default every supplied record is evaluated (`--val-fraction 0`). For
stride/tail or manifest-defined holdouts, supply the matching split arguments.
The same checkpoint can be evaluated on one domain or on the combined held-out
corpus. `--truncate` enables truncation to the configured context length.

## 4. Conditional SFT

SFT uses `direct_qwen`: molecular conditions are the same frozen BPE
tokens fed through Qwen's own embeddings. Text conditions use the separately
downloaded frozen `sentence-transformers/all-mpnet-base-v2` and a projection.
Raw FDDFS `<SEP>` separates multiple molecular components. Five fixed domain
tokens select target representation; adding a task does not add a task token.

An indexed SFT record contains a task, condition and encoded target. For example:

```python
from moltoken import MolTokenizer

record = {
    "id": "train-1",
    "task": "dna_diffusion_celltype_generation",
    "target_type": "dna",
    "condition": {"type": "text", "text": "K562", "molecule_bpe_ids": []},
    "target_bpe_ids": MolTokenizer().encode("ACGT", domain="dna"),
    "meta": {"split": "train"},
}
```

This short sequence illustrates the record format. `molecule_bpe_ids` and
`target_bpe_ids` contain raw IDs, without offset/BOS/EOS. The collator applies
framing, prefix masks and target-only loss.

Use the task materializers described in [benchmarks.md](benchmarks.md), or
`cli.data.materialize_encoded_biological_sft_root` for supported, already encoded
train/validation/test files. It writes the manifest and random-access
index. The task-to-dataset registry is in `pipeline/finetune/active_scope.py`;
register new tasks there without adding task-specific vocabulary entries.

For MOSES/GuacaMol, the source-reader CLI is
`cli.sft.preprocess_finetune_dataset`; then apply frozen BPE with
`moltoken compress-sft`. `cli.data.materialize_smallmol_train_holdout` creates
a training-only early-stopping partition from the two encoded training files.
Its hard links require source and destination on the same filesystem.

```bash
DATA=./data/sft BASE=./checkpoints/moltoken_0.55b \
  ENCODER=./checkpoints/all-mpnet-base-v2 TASK=unconditional_moses \
  OUT=./runs/moses BATCH=8 ACCUM=4 bash examples/sft.sh
```

Both stages run for up to 100 epochs with early stopping. Full training starts
from `alignment/best`. The validation split is derived from training data and
is used to select checkpoints; the benchmark test split is used for evaluation.
The trainer retains the final short batch and records the best and stopping epochs.

To generate from a task's indexed test conditions:

```bash
moltoken sample-sft --checkpoint ./runs/task/full/best \
  --data-root ./data/sft --text-encoder ./checkpoints/all-mpnet-base-v2 \
  --task dna_diffusion_celltype_generation --split test \
  --n-samples 32 --output ./samples.jsonl
```

MOSES and GuacaMol generate unconditionally through their evaluation adapters,
without an indexed SFT test split. Their sampling counts and metric commands
are documented in [Benchmark evaluation](benchmarks.md).
