"""Direct token-normalized held-out loss using the pretraining data path."""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from moltoken.checkpoints import load_pretrained
from cli.pretrain.train_qwen3_pretrain import (
    CompactIntOnlinePadDataset, collate_online_padded_samples, evaluate,
)
from cli.data.build_compact_int_index import expand_inputs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--token-files", nargs="+", required=True)
    p.add_argument("--index-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", choices=("fp32", "bf16"), default="fp32")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--context", type=int, default=512)
    p.add_argument("--truncate", action="store_true")
    p.add_argument("--val-fraction", type=float, default=0,
                   help="0 consumes every supplied held-out record; >0 selects the trainer's stride-held-out view")
    p.add_argument("--domain-split-manifest", type=Path)
    p.add_argument("--domain")
    a = p.parse_args()
    dtype = torch.bfloat16 if a.dtype == "bf16" else torch.float32
    model = load_pretrained(a.checkpoint, device=a.device, dtype=dtype)
    data = CompactIntOnlinePadDataset(
        paths=expand_inputs(a.token_files), index_dir=a.index_dir,
        split="val" if a.val_fraction or a.domain_split_manifest else "train",
        val_fraction=a.val_fraction, split_strategy="stride", token_offset=3,
        eos_token_id=2, pad_token_id=0, pad_length=a.context, truncate=a.truncate,
        seed=0, unfold_max_ratio=1, unfolder=None,
        domain_split_manifest=a.domain_split_manifest, domain_name=a.domain,
    )
    if not len(data):
        raise ValueError("held-out view is empty")
    loader = DataLoader(data, batch_size=a.batch_size, shuffle=False,
                        collate_fn=collate_online_padded_samples)
    result = evaluate(model, loader, torch.device(a.device), max_batches=0,
                       use_bf16=a.dtype == "bf16")
    result.update(checkpoint=str(a.checkpoint), context=a.context,
                  aggregation="sum(loss * predicted_tokens) / sum(predicted_tokens)",
                  records=len(data), domain=a.domain, complete=True)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
