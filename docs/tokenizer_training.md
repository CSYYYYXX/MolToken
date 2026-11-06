# FDDFS guide and integer BPE training

Existing checkpoints use the bundled frozen codebook. This page describes
training a new tokenizer and applying an ordered merge table to a corpus.

## Guide and serialization

After `cli.data.build_heavy_atom_graphs`, run:

```bash
python -m cli.pretrain.serialize_fddfs_full \
  --graph-root ./data/graphs_heavy_atom_normv2_compact_v1 \
  --out-dir ./data/new_fddfs --phase all --split train --workers 8
```

`--phase count` and `--phase serialize` can be run separately. The guide is
learned from training graphs. The serializer outputs compact-int base-token
shards and a manifest. `tools/yttm_int_bpe/tools/build_fddfs_int_shards.py`
can repartition that manifest while preserving record boundaries.

## Native toolchain

```bash
bash tools/yttm_int_bpe/tools/build_int_bpe_bench.sh
bash tools/yttm_int_bpe/tools/build_prefix_merge_int_shard.sh

tools/yttm_int_bpe/build/int_bpe_bench \
  --mode single --input ./data/base_tokens.bin --out-dir ./data/new_codebook \
  --num-merges 16000 --min-frequency 80000 --threads 8
```

`int_bpe_bench` is the retained production binary name, not a timing-only
benchmark. It supports single-process, filesystem-distributed (`dist`) and
TCP-distributed (`dist-tcp`) training, ordered application and compact output.
Choose `--reserve-pairs` for the dataset; position indexes can consume much
more memory than the on-disk token array. For distributed operation supply
`--rank`, `--world-size`, and `--tcp-rendezvous` (TCP mode) on allocated hosts.
Each rank reads its shard with `--input-shard`.

Application of a complete codebook:

```bash
tools/yttm_int_bpe/build/int_bpe_bench \
  --mode apply-encode --input ./data/base_tokens.bin --out-dir ./data/apply_log \
  --apply-merges ./data/new_codebook/merges.csv --threads 8 --no-rle \
  --encoded-output ./data/encoded.bin --encoded-compact-int \
  --token-offset 0 --no-eos
```

The encoded file stores raw IDs; the training collator adds model offset and
BOS/EOS markers. Each input header declares the vocabulary's maximum ID, shared
across shards even when individual shards do not contain every base symbol.

## Fixed merge prefixes

A high-frequency prefix can be applied first to reduce the number of tokens
and pair-position records before training a suffix. The original frozen
codebook used a 70-merge prefix. The prefix IDs and rank order are preserved:

```bash
tools/yttm_int_bpe/build/prefix_merge_int_shard \
  --input ./data/base_tokens.bin --output ./data/prefix.bin \
  --merges ./data/prefix_codebook/merges.csv --limit 70

tools/yttm_int_bpe/build/int_bpe_bench \
  --mode single --input ./data/prefix.bin --out-dir ./data/suffix_codebook \
  --num-merges 15930 --min-frequency 80000 --threads 8

python tools/yttm_int_bpe/tools/concat_prefix_suffix_merges.py \
  --prefix ./data/prefix_codebook/merges.csv --prefix-limit 70 \
  --suffix ./data/suffix_codebook/merges.csv --out ./data/full_merges.csv
```

When applying the combined table to an already prefix-compressed file, pass
`--apply-skip 70` to `int_bpe_bench`. Apply the full table without that option
when starting from base tokens.
