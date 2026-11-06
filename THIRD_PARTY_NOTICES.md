# Third-party components

MolToken's project code is released under MIT. This does not relicense external
datasets, model weights, evaluation resources, or upstream dependencies.

- Integer BPE tooling retains the YouTokenToMe MIT notice (VK.com, 2019) in
  `tools/yttm_int_bpe/LICENSE`.
- `flat_hash_map.h` retains the Boost Software License 1.0; see
  `tools/yttm_int_bpe/youtokentome/cpp/third_party/LICENSE` and its header.
- The Qwen3 implementation is imported from Hugging Face Transformers, not
  copied into this repository. Install its published package and retain its
  Apache-2.0 license when redistributing dependencies.
- RDKit, PyTorch, xFormers, MOSES, GuacaMol, ESM, RefineGNN, TCR-TRANSLATE and
  other external evaluators remain separately distributed dependencies with
  their respective licenses. Their weights, datasets and repositories are not
  bundled here.

The frozen FDDFS guide, vocabulary and integer merge rules in
`moltoken/assets/fddfs_bpe` are included to preserve the tokenizer used by the
released checkpoints. They do not contain the pretraining records.
