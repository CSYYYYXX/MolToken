# Source release validation — 2026-11-07

Validation ran on Linux x86_64 with Python 3.11 on an allocated CPU worker.

- 235 unit/regression tests passed.
- 72 production CLI help entry points loaded successfully.
- Direct local import closure and referenced project-script checks passed.
- Production FDDFS, BPE and the separate rebuttal ordering extension built.
- Rebuttal frequency/fixed-order C++ outputs matched Python and graph round trips passed.
- Rebuttal compact integer-BPE training matched the serial oracle on 17 synthetic sequences; applying the learned merges reproduced the expected token count.
- Eight-step synthetic integration passed: encoding, indexing, pretraining, held-out loss, SFT materialization, Alignment, Full and sampling. Full used the matching Alignment checkpoint.
- The publication candidate passed Python parsing, shell syntax, source credential-pattern and unrelated-project checks. No model tensors, private migration documents or storage catalogs are included.

## Environment versions

- torch: 2.8.0
- transformers: 4.57.6
- numpy: 1.26.4
- rdkit: 2023.9.6
- safetensors: 0.6.2
- accelerate: 1.10.1
- pybind11: 3.0.3
- numba: 0.68.1
- pyarrow: 21.0.0
- pandas: 2.2.3
- pytest: 8.4.2
- orjson: 3.13.0
- zstandard: 0.25.0
- biopython: 1.88

These are validation versions, not a claim that every CUDA stack is compatible.
The `train` extra pins Transformers 4.57.6; choose torch/xFormers together for GPU work.

## Limits

CPU synthetic tests establish source execution and the tested semantic invariants,
not scientific benchmark scores. No full-corpus training, GPU packed-attention
campaign, multi-node run or full official evaluator campaign was repeated for
publication. External data, tokenizer variants, pretrained weights and evaluator
assets must match each experiment's configuration and lineage. Historical
research failures remain failures; the four incomplete transfer runs are not
certified by these CPU checks.

The main model/codec implementation was adapted from the established MolToken
source release; research drivers include the later E2/E3 and transfer campaign.
Portability changes replace site paths and add missing build/test dependencies;
they do not intentionally alter model parameter names, token IDs or training
protocols. See [research source map](research.md).
