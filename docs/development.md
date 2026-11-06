# Development

## Source layout

| Directory | Contents |
|---|---|
| `moltoken/` | Public tokenizer and checkpoint APIs, command dispatcher, frozen tokenizer assets |
| `pipeline/graph/` | Heavy-atom graph construction and residue templates |
| `pipeline/serialization/` | FDDFS serialization and inverse decoding |
| `pipeline/finetune/` | Conditional model, collators and task registry |
| `cli/data/` | Download, normalization, indexing and task materialization |
| `cli/pretrain/` | Serialization, autoregressive training and loss evaluation |
| `cli/sft/` | SFT preprocessing, BPE compression and two-stage training |
| `cli/evaluate/` | Sampling and task-specific evaluators |
| `src/algorithms/compression/` | Python and native integer BPE implementations |
| `tools/yttm_int_bpe/` | Large-corpus BPE training and application tools |
| `configs/`, `examples/`, `tests/` | Configurations, usage examples and regression tests |

The `pipeline`, `cli` and `src` namespaces are retained for compatibility with
existing checkpoints and training code. The native FDDFS extension retains
its historical filename, `_stackk_cpp`.

## Tests

From the repository root:

```bash
python -m pip install -e '.[train,data,test]'
python -m pytest
```

The suite covers graph and sequence round trips, token IDs, checkpoint loading,
collation, training-data accounting and evaluator helpers. Tests use local
synthetic fixtures and do not require training datasets or pretrained weights.
Benchmark scoring additionally requires the resources listed in
[Benchmark evaluation](benchmarks.md).
