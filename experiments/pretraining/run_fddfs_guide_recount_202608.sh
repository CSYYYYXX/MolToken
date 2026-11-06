#!/usr/bin/env bash
set -u
# Phase 0.2: 重跑 count 阶段，拿到与预训练一致的语料级 FDDFS guide。
#
# 为什么要重跑：guide 决定 FDDFS 的 DFS 遍历顺序，而 transition_key_from_features
# 的排序主键是 triplet_counts。旧的 guide_counts.json 只存了 triplet_unique（计数），
# 没存字典本身，无法复现。Phase 0.1 已给 save_count_payload 补上落盘。
#
# 纯 CPU，不碰 GPU。上次实测 4447 秒 / 64 workers / 169,452 文件 / C++ 后端。
REPO=./scaling-csy
PIPE=$REPO/code/pipeline
export PYTHONPATH=$PIPE
DATA=./storage/molscaling/data
GRAPH=$DATA/graphs_heavy_atom_normv2_compact_v1
OUT=$DATA/fddfs_guide_recount_v1
SCRIPT=$PIPE/cli/pretrain/serialize_fddfs_full.py
WORKERS=${WORKERS:-64}

PY=/data/envs/scaling-csy/bin/python
[ -x "$PY" ] || PY=${PYTHON:-python}
echo "=== PY=$PY WORKERS=$WORKERS ==="
# 注意：读 *.pkl.zst 走的是 zstd **二进制**（pipeline.graph.io.find_zstd），
# 不是 python 的 zstandard 模块，别再检查错对象了。
"$PY" -c "import numpy; print('numpy', numpy.__version__)" || { echo "NO_NUMPY"; exit 1; }
"$PY" -c "
import sys; sys.path.insert(0, '$PIPE')
from pipeline.graph.io import find_zstd
print('zstd:', find_zstd())
" || { echo "NO_ZSTD"; exit 1; }
[ -d "$GRAPH" ] || { echo "NO_GRAPH_ROOT: $GRAPH"; exit 1; }

echo "=== graph root ==="
ls "$GRAPH" | head -5
echo "  domain 目录数: $(ls $GRAPH | wc -l)"
date '+%F %T %Z  开始'

# --split 必须写全！脚本默认只跑 train（167,788 文件），而预训练那次跑的是
# train,valid,test（169,452 文件）。少算 valid+test 会让 triplet_counts 差 30 条、
# 共享三元组计数差约 1%，足以让部分邻居排序翻转 —— 那就复现不了预训练 guide。
"$PY" "$SCRIPT" \
  --graph-root "$GRAPH" \
  --out-dir "$OUT" \
  --phase count \
  --split train,valid,test \
  --workers "$WORKERS" \
  --log-every-files 5000 2>&1 | tail -60

RC=${PIPESTATUS[0]}
date '+%F %T %Z  结束'
echo "COUNT_RC=$RC"

# ---- 验收：triplet_counts 必须落盘，且与旧 guide 的 node/edge 规模一致 ----
"$PY" - <<'PYEOF'
import json, pathlib
new = pathlib.Path("./storage/molscaling/data/fddfs_guide_recount_v1/guide_counts.json")
old = pathlib.Path("./storage/molscaling/data/fddfs_heavy_atom_normv2_compact_v1/guide_counts.json")
if not new.exists():
    print("VERIFY_FAIL: 新 guide_counts.json 不存在"); raise SystemExit(1)
n = json.loads(new.read_text())
ok = True
# 文件数必须是 169,452（train+valid+test）。只跑 train 是 167,788，那样复现不了
# 预训练 guide —— 第一次就栽在这，加个硬检查。
summ = new.parent / "count_summary.json"
if summ.exists():
    s = json.loads(summ.read_text())
    nf, splits = s.get("files"), sorted(s.get("split_stats", {}))
    print(f"  files={nf:,}  splits={splits}")
    if nf != 169452:
        print(f"VERIFY_FAIL: 文件数 {nf:,} != 169,452，--split 是不是漏了？"); ok = False
    if splits != ["test", "train", "valid"]:
        print(f"VERIFY_FAIL: split 不全，实得 {splits}"); ok = False
if "triplet_counts" not in n:
    print("VERIFY_FAIL: triplet_counts 没落盘"); ok = False
else:
    nt, nu = len(n["triplet_counts"]), n.get("triplet_unique")
    print(f"  triplet_counts={nt:,}  triplet_unique={nu:,}  一致={nt==nu}")
    ok &= (nt == nu)
print(f"  node_counts={len(n.get('node_counts',{})):,}  edge_counts={len(n.get('edge_counts',{})):,}")
if old.exists():
    o = json.loads(old.read_text())
    for k in ("node_counts", "edge_counts"):
        a, b = len(n.get(k, {})), len(o.get(k, {}))
        print(f"  {k}: 新={a} 旧={b} {'✓' if a==b else '✗ 与旧 guide 规模不符'}")
        ok &= (a == b)
    a, b = len(n.get("triplet_counts", {})), o.get("triplet_unique")
    print(f"  triplet: 新={a} 旧记录={b} {'✓' if a==b else '✗'}")
    ok &= (a == b)
print("VERIFY_OK" if ok else "VERIFY_FAIL")
PYEOF

echo
ls -la "$OUT" 2>&1
echo GUIDE_RECOUNT_DONE
