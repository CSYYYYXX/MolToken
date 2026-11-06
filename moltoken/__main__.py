"""Portable command dispatcher; no cluster allocation or implicit downloads."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import runpy
import shutil
import sys

COMMANDS = {
    "pretrain": "cli.pretrain.train_qwen3_pretrain",
    "sft": "cli.sft.train_axis_b_repeated_stage_sft",
    "serialize": "cli.pretrain.serialize_fddfs_full",
    "compress-sft": "cli.sft.compress_finetune_fddfs_bpe",
    "sample-sft": "cli.evaluate.sample_axis_b_sft",
    "decode": "cli.evaluate.eval_axis_b_samples",
    "moses": "cli.evaluate.evaluate_axis_b_moses_official",
    "guacamol": "cli.evaluate.evaluate_axis_b_guacamol_official",
    "evaluate-loss": "moltoken.evaluate_loss",
    "encode-corpus": "moltoken.encode_corpus",
}


def main():
    argv = sys.argv[1:]
    if argv and argv[0] in COMMANDS:
        module = COMMANDS[argv[0]]
        sys.argv = [module, *argv[1:]]
        runpy.run_module(module, run_name="__main__")
        return
    parser = argparse.ArgumentParser(description="MolToken: graph -> FDDFS -> BPE -> Transformer")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in COMMANDS:
        sub.add_parser(name, help=f"see moltoken {name} --help")
    encode = sub.add_parser("encode", help="encode one SMILES or biological sequence")
    encode.add_argument("sequence")
    encode.add_argument("--domain", choices=("small_molecule", "peptide", "protein", "dna", "rna"), default="small_molecule")
    encode.add_argument("--model-ids", action="store_true")
    assets = sub.add_parser("assets", help="install frozen assets under the historical data layout")
    assets.add_argument("--data-root", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == "encode":
        from moltoken import MolTokenizer
        print(json.dumps(MolTokenizer().encode(args.sequence, args.domain, model_ids=args.model_ids)))
    elif args.command == "assets":
        from pipeline.paths import TOKENIZER_ASSET_ROOT
        destinations = {
            "guide_counts.json": "fddfs_guide_recount_v1/guide_counts.json",
            "vocab.json": "fddfs_guide_recount_v1/vocab.json",
            "merges.csv": "fddfs_bpe_full_prefix70_64k_mf80000_v1/codebook_fddfs_prefix70_mf80000_v1/merges.csv",
        }
        for name, relative in destinations.items():
            path = args.data_root / relative
            source = TOKENIZER_ASSET_ROOT / name
            if path.exists():
                if path.read_bytes() != source.read_bytes():
                    raise FileExistsError(f"refusing to overwrite a different tokenizer asset: {path}")
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, path)
            print(path)


if __name__ == "__main__":
    main()
