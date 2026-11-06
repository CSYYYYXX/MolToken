#!/usr/bin/env python3
"""C4: 在干净的 stride-val 留出集上，用 67 个 final checkpoint 补算真正的留出 loss。

关键点：本脚本**直接导入训练脚本** train_qwen3_pretrain.py，复用它的
  - CompactIntOnlinePadDataset(split="val")   验证集切分
  - collate_online_padded_samples             collate
  - evaluate()                                token 加权 loss
  - PackedXformersQwen3ForCausalLM            模型实现
因此留出 loss 与训练时报告的 training loss 在**完全相同的协议**下计算，可直接对照。

验证集对所有 run 完全一致（同一 compact shards、stride=1000 切分、seed 42、
pad 512、truncate=True、无 BPE unfold），故每个 worker 只构建一次 val dataset，
在其分到的所有 checkpoint 上复用。

幂等：若某 run 的 out_dir 下已存在 holdout_loss.json 则跳过，可安全重跑。

用法（在带 xformers 的 GPU worker 内、用训练同款 conda env 运行）：
    python eval_c4_holdout_loss.py --runs-file <file with one out_dir per line> \
        [--eval-batch-size 64] [--max-batches -1] [--out-name holdout_loss.json]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

_TRAIN_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "cli"
    / "pretrain"
    / "train_qwen3_pretrain.py"
)


def load_train_module():
    spec = importlib.util.spec_from_file_location("train_qwen3_pretrain", _TRAIN_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_val_dataset(tm, cfg, device):
    """用训练同款参数构建 val 数据集（对所有 run 一致，只需构建一次）。

    数据集相关参数在 run_config.json 里嵌套于 cfg["args"]（顶层只有
    dataset_mode / model_implementation 等少数镜像键）。
    """
    a = cfg["args"]
    # val_token_files 用 run 自己记录的 glob，避免硬编码路径漂移
    val_token_globs = a.get("val_token_files") or a.get("token_files")
    token_paths = tm.expand_paths([str(p) for p in val_token_globs])
    index_dir = Path(a["compact_index_dir"])
    val_ds = tm.CompactIntOnlinePadDataset(
        token_paths,
        index_dir,
        "val",
        float(a["val_fraction"]),
        a["split_strategy"],
        int(a["token_offset"]),
        int(a["eos_token_id"]),
        int(a["pad_token_id"]),
        int(a["online_pad_length"]),
        True,            # val 始终 truncate（与训练 val_ds 一致）
        int(a["seed"]),
        4.0,             # unfold_max_ratio：val 无 unfolder，占位
        None,            # unfolder：val 恒为 None
        None,            # domain_id_dir：不影响 loss / 样本选择
    )
    return val_ds


def load_model(tm, final_dir: Path, device):
    from transformers import Qwen3Config
    config = Qwen3Config.from_pretrained(final_dir)
    config.use_cache = False
    model = tm.PackedXformersQwen3ForCausalLM(config)
    state = torch.load(final_dir / "pytorch_model.bin", map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    # tie_word_embeddings=True：lm_head.weight 与 embed_tokens.weight 同一张量，
    # strict=False 容许其中一个 key 缺失；两者都在时按原样加载。
    hard_missing = [k for k in missing if "lm_head" not in k]
    if hard_missing or unexpected:
        raise RuntimeError(f"state_dict mismatch for {final_dir}: missing={hard_missing} unexpected={unexpected}")
    model.to(device)
    model.eval()
    return model, config


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-file", type=Path, required=True, help="每行一个 run 的 out_dir")
    ap.add_argument("--eval-batch-size", type=int, default=64)
    ap.add_argument("--max-batches", type=int, default=-1, help="-1 表示整个 val 集")
    ap.add_argument("--out-name", default="holdout_loss.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("packed-xformers 需要 GPU（xformers），请在 GPU worker 内运行")

    tm = load_train_module()

    run_dirs = [Path(line.strip()) for line in args.runs_file.read_text().splitlines() if line.strip()]
    print(f"[c4] {len(run_dirs)} runs to evaluate on this worker", flush=True)

    val_ds = None
    val_loader = None
    done, skipped, failed = 0, 0, 0

    for i, out_dir in enumerate(run_dirs):
        out_json = out_dir / args.out_name
        if out_json.is_file():
            print(f"[c4] ({i+1}/{len(run_dirs)}) skip (exists): {out_dir}", flush=True)
            skipped += 1
            continue

        final_dir = out_dir / "final"
        run_config_path = out_dir / "run_config.json"
        if not (final_dir / "pytorch_model.bin").is_file() or not run_config_path.is_file():
            print(f"[c4] ({i+1}/{len(run_dirs)}) MISSING ckpt/config: {out_dir}", flush=True)
            failed += 1
            continue

        cfg = json.loads(run_config_path.read_text())
        summary_path = out_dir / "summary.json"
        summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
        reported_training_loss = summary.get("final_logged_train_loss")

        # val dataset 对所有 run 一致，只构建一次
        if val_ds is None:
            t0 = time.time()
            val_ds = build_val_dataset(tm, cfg, device)
            val_loader = DataLoader(
                val_ds,
                batch_size=args.eval_batch_size,
                shuffle=False,
                collate_fn=tm.collate_online_padded_samples,
                num_workers=0,
                pin_memory=True,
                drop_last=False,
            )
            print(f"[c4] val dataset built: {len(val_ds)} samples in {time.time()-t0:.1f}s", flush=True)

        try:
            t0 = time.time()
            model, config = load_model(tm, final_dir, device)
            res = tm.evaluate(model, val_loader, device, args.max_batches, use_bf16=True)
            elapsed = time.time() - t0

            payload = {
                "out_dir": str(out_dir),
                "holdout_loss": res["loss"],
                "holdout_ppl": res["ppl"],
                "holdout_tokens": res["tokens"],
                "val_samples": len(val_ds),
                "eval_batch_size": args.eval_batch_size,
                "max_batches": args.max_batches,
                "reported_training_loss": reported_training_loss,
                "parameter_count": cfg.get("parameter_count"),
                "protocol": "identical to training val: stride-1000 split, pad512, truncate, seed42",
                "elapsed_sec": elapsed,
            }
            out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[c4] ({i+1}/{len(run_dirs)}) {out_dir.parent.name}/{out_dir.name} "
                  f"holdout_loss={res['loss']:.6f} tokens={res['tokens']} ({elapsed:.1f}s)", flush=True)
            done += 1
            del model
            torch.cuda.empty_cache()
        except Exception as exc:  # noqa: BLE001
            print(f"[c4] ({i+1}/{len(run_dirs)}) FAILED {out_dir}: {exc}", flush=True)
            failed += 1

    print(f"[c4] worker done: {done} evaluated, {skipped} skipped, {failed} failed", flush=True)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
