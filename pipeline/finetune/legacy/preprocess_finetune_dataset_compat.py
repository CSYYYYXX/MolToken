#!/usr/bin/env python3
"""Streaming SFT preprocessor with retired reader implementations retained.

每个数据集只需写一个 reader（产出 ``RawSample``），编码/并行/验收/落盘全共用。

当前支持的数据集（``--dataset``）：

  unconditional   benchmark/clean/{moses,guacamol}/*.jsonl   无条件生成

本文件暂时保留若干历史 reader 的实现，便于追溯旧记录格式，但命令行不再暴露
这些退休 benchmark；若需重新启用，必须先显式恢复任务范围和官方评测协议。

**必须在 worker 上跑**：需要 rdkit，且是 CPU 密集操作。跳板机资源很小，
OOM 会连带搞死正在跑的训练；launcher 应先做机器规格检查。
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import re
import sys
import time
from collections import Counter, deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

PIPELINE_ROOT = Path(__file__).resolve().parents[3]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from pipeline.finetune.fddfs_encode import (  # noqa: E402
    DEFAULT_GUIDE_PATH,
    CorpusGuide,
    FastCorpusEncoder,
    RingIdOverflow,
    UnknownFeatureError,
    ValidationError,
    base_vocab_tokens,
    encode_molecule,
    encode_molecule_fast,
    filter_unknown_token_records,
    load_corpus_guide,
    make_record,
    unknown_tokens_in,
    validate_encoded,
)
from pipeline.finetune.active_scope import (  # noqa: E402
    ACTIVE_UNCONDITIONAL_BENCHMARKS,
)

csv.field_size_limit(10 * 1024 * 1024)

# FDDFS 的纯 Python 回退路径（serialize_fddfs / SequenceParser）是递归实现，
# 深度 O(N)。蛋白 1024 残基 ≈ 8000 重原子，默认 limit=1000 必然 RecursionError。
# 仓库里其它 12 个序列化脚本都设了这一行，唯独本脚本漏了 —— 补上。
sys.setrecursionlimit(max(sys.getrecursionlimit(), 1_000_000))


@dataclass
class RawSample:
    """reader 的统一产出。编码前的原始形态。

    ``condition_type`` 单列一个字段是必要的：跨模态任务里 condition 和 target
    可以是不同模态（例如 pocket→ligand 的 condition 是蛋白、target 是小分子），
    用同一个 target_type 去编码 condition 会把蛋白序列当 SMILES 解析。
    不填时默认与 ``target_type`` 相同（同模态任务，如逆合成）。
    """

    sample_id: str
    task: str
    target_type: str
    target_smiles: str | None = None
    target_sequence: str | None = None
    condition_text: str | None = None
    condition_smiles: str | None = None
    condition_sequence: str | None = None
    condition_type: str | None = None
    condition_role: str | None = None
    condition_instruction: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.condition_type is None:
            self.condition_type = self.target_type


# ---------------------------------------------------------------------------
# readers
# ---------------------------------------------------------------------------


def read_unconditional(path: Path, *, benchmark: str) -> Iterator[RawSample]:
    """benchmark/clean/<name>/{train,valid,test}.jsonl → 无条件生成。

    这批文件已经是统一 schema，带 ``target_type`` / ``canonical_smiles`` / ``sequence``。
    """

    if benchmark not in ACTIVE_UNCONDITIONAL_BENCHMARKS:
        raise ValueError(
            f"unsupported formal unconditional benchmark {benchmark!r}; "
            f"expected one of {ACTIVE_UNCONDITIONAL_BENCHMARKS}"
        )
    with path.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            smi = rec.get("canonical_smiles") or rec.get("smiles")
            seq = rec.get("sequence")
            if not smi and not seq:
                continue
            yield RawSample(
                sample_id=rec["id"],
                task=f"unconditional_{benchmark}",
                target_type=rec.get("target_type", "small_molecule"),
                target_smiles=smi,
                target_sequence=seq,
                meta={"benchmark": benchmark},
            )


# OpenMolIns / TOMG-Bench 的 instruction 里内嵌输入分子，形如
#   "Please add a hydroxyl to the molecule CCN(CC)C(=O)CNc1nc(C(N)=O)ccc1N."
#   "Modify the molecule C[C@H]1C... to increase its QED value."
_MOL_IN_TEXT = re.compile(r"the molecule\s+(\S+)")


def extract_input_smiles(instruction: str) -> str | None:
    """从 instruction 里抠出输入分子。

    尾部句号有歧义——SMILES 里的 ``.`` 是断开组分分隔符（如 ``CC.CC``），
    所以不能无脑 strip。策略：先试原样，再试去掉尾部句号，
    哪个能被 RDKit 解析就用哪个。都不行返回 None。
    """

    from rdkit import Chem
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")

    m = _MOL_IN_TEXT.search(instruction)
    if not m:
        return None
    cand = m.group(1)
    for trial in (cand, cand.rstrip(".")):
        if trial and Chem.MolFromSmiles(trial) is not None:
            return trial
    return None


#: OpenMolIns 子任务 → (任务族, 是否有输入分子)
_OPENMOLINS_SUBTASKS = {
    "AtomNum": ("molcustom", False),
    "BondNum": ("molcustom", False),
    "FunctionalGroup": ("molcustom", False),
    "AddComponent": ("moledit", True),
    "DelComponent": ("moledit", True),
    "SubComponent": ("moledit", True),
    "LogP": ("molopt", True),
    "MR": ("molopt", True),
    "QED": ("molopt", True),
}


def read_openmolins(path: Path, *, benchmark: str = "openmolins") -> Iterator[RawSample]:
    """OpenMolIns train.csv → 文本条件 / 文本+分子双条件生成。

    列 ``SubTask / Instruction / molecule``，其中 ``molecule`` 是**输出**（目标）；
    编辑/优化类的输入分子内嵌在 ``Instruction`` 里，需要抠出来做 molecule condition。
    MolCustom 三个子任务没有输入分子，只有文本条件。
    """

    with path.open(encoding="utf-8", newline="") as f:
        for i, row in enumerate(csv.DictReader(f)):
            sub = row.get("SubTask", "").strip()
            family, has_input = _OPENMOLINS_SUBTASKS.get(sub, (None, False))
            if family is None:
                continue
            target = (row.get("molecule") or "").strip()
            instruction = (row.get("Instruction") or "").strip()
            if not target or not instruction:
                continue
            cond_smi = extract_input_smiles(instruction) if has_input else None
            if has_input and cond_smi is None:
                continue  # 抠不出输入分子的丢弃，宁缺毋滥
            yield RawSample(
                sample_id=f"{benchmark}_{sub}_{i}",
                task=f"{family}_{sub.lower()}",
                target_type="small_molecule",
                target_smiles=target,
                condition_text=instruction,
                condition_smiles=cond_smi,
                meta={"subtask": sub, "family": family, "benchmark": benchmark},
            )


def read_chebi20(path: Path, *, benchmark: str = "chebi20") -> Iterator[RawSample]:
    """ChEBI-20 → 文本条件分子生成。

    注意后缀是 .csv 但实际是 **tab 分隔**，列 ``CID / SMILES / description``。
    """

    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            smi = (row.get("SMILES") or "").strip()
            desc = (row.get("description") or "").strip()
            if not smi or not desc:
                continue
            yield RawSample(
                sample_id=f"chebi20_{row.get('CID','?')}",
                task="text2mol",
                target_type="small_molecule",
                target_smiles=smi,
                condition_text=desc,
                meta={"cid": row.get("CID"), "benchmark": benchmark},
            )


def read_tomgbench(path: Path, *, benchmark: str = "tomgbench") -> Iterator[RawSample]:
    """S2-TOMG-Bench 的单个子任务 CSV → 评测样本。

    这是**评测集**，三族的列结构不同（实测确认，别照抄 OpenMolIns 的假设）：

    - ``MolEdit_*``：``index, Instruction, molecule, <extra>``
      ``molecule`` 是**输入**分子（与 instruction 内嵌的相同），无 ground truth。
    - ``MolOpt_*``：``index, Instruction, molecule, <property>``，同上。
    - ``MolCustom_*``：``Instruction, carbon, oxygen, nitrogen, ...``
      **没有 index 也没有 molecule 列** —— 纯生成任务，约束条件就是那些计数列，
      靠规则检查器数生成结果里的原子/键/官能团来评分。

    三族都没有参考答案，所以 ``target`` 一律留空、由采样阶段生成。
    约束/属性列原样存进 meta，供评测脚本使用。
    """

    sub = path.stem  # 例如 MolEdit_AddComponent
    family = sub.split("_")[0].lower()
    with path.open(encoding="utf-8", newline="") as f:
        for i, row in enumerate(csv.DictReader(f)):
            instruction = (row.get("Instruction") or "").strip()
            if not instruction:
                continue

            cond_smi: str | None = None
            if family != "molcustom":
                cond_smi = (row.get("molecule") or "").strip() or extract_input_smiles(instruction)
                if cond_smi is None:
                    continue  # 编辑/优化类拿不到输入分子就没意义

            extra = {k: v for k, v in row.items()
                     if k not in ("index", "Instruction", "molecule")}
            yield RawSample(
                sample_id=f"tomg_{sub}_{row.get('index', i)}",
                task=f"{family}_{sub.split('_', 1)[1].lower()}",
                target_type="small_molecule",
                target_smiles=None,          # 评测集无 ground truth
                condition_text=instruction,
                condition_smiles=cond_smi,
                meta={"subtask": sub, "benchmark": benchmark,
                      "split": "test", "constraints": extra},
            )


def read_swissprotclap(
    path: Path,
    *,
    benchmark: str = "swissprotclap",
    max_residues: int = 1024,
) -> Iterator[RawSample]:
    """SwissProtCLAP parquet → 文本条件蛋白质设计。

    列 ``UniProt ID / Protein Sequence / Functional Description``，两分片约 54 万条。

    ``max_residues`` 截断：蛋白越长图越大、序列化越慢，而且超长序列（>1024 残基）
    在文本条件设计任务里本来就少见。默认 1024 残基（约 8000 重原子）。
    传 0 取消限制。被过滤掉的条数会打印出来。
    """

    import pyarrow.parquet as pq

    files = sorted(path.glob("*.parquet")) if path.is_dir() else [path]
    if not files:
        raise FileNotFoundError(f"{path} 下没有 parquet")

    cols = ["UniProt ID", "Protein Sequence", "Functional Description"]
    n_long = 0
    for fp in files:
        for row in _iter_parquet_rows(pq.ParquetFile(fp), cols, batch_size=1024):
            seq = (row.get("Protein Sequence") or "").strip().upper()
            desc = (row.get("Functional Description") or "").strip()
            if not seq or not desc:
                continue
            if max_residues and len(seq) > max_residues:
                n_long += 1
                continue
            yield RawSample(
                sample_id=f"swissprot_{row.get('UniProt ID','?')}",
                task="text2protein",
                target_type="protein",
                target_sequence=seq,
                condition_text=desc,
                meta={"uniprot": row.get("UniProt ID"), "benchmark": benchmark,
                      "n_residues": len(seq)},
            )
    if n_long:
        print(f"    (跳过 {n_long:,} 条超过 {max_residues} 残基的序列)", file=sys.stderr)


def _iter_parquet_rows(pf: Any, columns: list[str], batch_size: int = 256) -> Iterator[dict[str, Any]]:
    """按批流式读 parquet，逐行产出 dict。

    整表 ``read_table().to_pylist()`` 会把所有列一次性物化成 Python 对象，
    对含大数组列的文件（如 PDBBind 的坐标）既慢又吃内存。
    """

    for batch in pf.iter_batches(batch_size=batch_size, columns=columns):
        yield from batch.to_pylist()


def _as_xyz(raw: Any, label: str, sample_id: str) -> "Any":
    """把 parquet 里的坐标字段整成 (N, 3) 的 numpy 数组。

    HF 的 Sequence 类型落地后可能是嵌套 list[[x,y,z], ...] 也可能是拍平的
    [x,y,z,x,y,z,...]，两种都兼容；形状对不上就报错，不猜。
    """

    import numpy as np

    arr = np.asarray(raw, dtype=np.float64)
    if arr.ndim == 2 and arr.shape[1] == 3:
        return arr
    if arr.ndim == 1 and arr.size % 3 == 0:
        return arr.reshape(-1, 3)
    raise ValueError(f"{sample_id}: {label} 形状无法解析为 (N,3)，实得 {arr.shape}")


def _finite_xyz_rows(xyz: "Any") -> tuple["Any", "Any"]:
    """Return finite coordinate rows and their boolean mask.

    ``jglaser/pdbbind_complexes`` aligns ligand coordinates to tokenized
    SMILES. Punctuation/non-atom tokens deliberately carry NaN rows; those
    rows are masked out instead of making the whole ligand record invalid.
    """

    import numpy as np

    arr = np.asarray(xyz, dtype=np.float64)
    mask = np.isfinite(arr).all(axis=1)
    return arr[mask], mask


def _select_pocket_indices(
    sequence: str,
    receptor_xyz: "Any",
    ligand_xyz: "Any",
    *,
    pocket_cutoff: float,
    min_pocket_residues: int,
) -> tuple["Any", int, int, str]:
    """Select pocket residues after removing masked ligand-token rows.

    Returns ``(indices, ligand_atom_rows, masked_ligand_rows, reason)``.
    ``reason`` is empty on success and otherwise records why no pocket was
    produced. Coordinates are used only here during preprocessing; they are
    never included in the model record.
    """

    import numpy as np

    rec = np.asarray(receptor_xyz, dtype=np.float64)
    ligand_atom_xyz, ligand_mask = _finite_xyz_rows(ligand_xyz)
    n_atom_rows = int(ligand_atom_xyz.shape[0])
    n_masked_rows = int((~ligand_mask).sum())

    if rec.shape[0] != len(sequence) or not bool(np.isfinite(rec).all()):
        return np.empty((0,), dtype=np.int64), n_atom_rows, n_masked_rows, (
            "receptor_xyz_not_finite_or_sequence_length_mismatch"
        )
    if n_atom_rows == 0:
        return np.empty((0,), dtype=np.int64), 0, n_masked_rows, (
            "no_finite_ligand_atom_coords"
        )

    distance = np.linalg.norm(
        rec[:, None, :] - ligand_atom_xyz[None, :, :], axis=-1
    ).min(axis=1)
    indices = np.nonzero(distance < pocket_cutoff)[0]
    if indices.size < min_pocket_residues:
        return indices, n_atom_rows, n_masked_rows, "fewer_than_min_pocket_residues"
    return indices, n_atom_rows, n_masked_rows, ""


def read_pdbbind(
    path: Path,
    *,
    benchmark: str = "pdbbind",
    pocket_cutoff: float = 10.0,
    min_pocket_residues: int = 5,
    max_residues: int = 1024,
    require_pocket: bool = True,
    stats: dict[str, int] | None = None,
) -> Iterator[RawSample]:
    """PDBBind 复合物 → 靶点条件的配体生成。

    列 ``name / seq / smiles / receptor_xyz / ligand_xyz / ...``，16,079 条。

    **坐标语义（重要）**：``ligand_xyz`` 是按 Schwaller SMILES token 对齐的，
    非原子 token（括号、键符号、环编号等）按设计写成 NaN；只有这些行应被
    屏蔽，不能用 ``np.isfinite(ligand_xyz).all()`` 把整条记录判为无坐标。
    ``receptor_xyz`` 则是每个残基的 C-alpha 坐标。

    因此本 reader 有两种模式，按每条记录的坐标可用性自动选择：

    - **pocket 模式**（坐标有效）：口袋 = Cα 到配体任意原子距离 < ``pocket_cutoff`` Å
      的残基，按原序列顺序拼成伪肽链。task = ``pocket2ligand``。
    - **sequence 模式**（坐标形状/受体坐标异常，或没有有限的配体原子坐标）：
      仅在 ``require_pocket=False`` 时用**全蛋白序列**做条件。正式 pocket benchmark
      默认跳过这类记录，避免在同一产物中混入 ``protein2ligand``。

    **不要把 sequence 模式的结果当成 pocket-conditioned 来报**——它是靶点条件的
    配体生成，不等价于 LOGOS 的口袋条件设定，Vina 数字不可直接横比。当前快照的
    有限配体原子坐标足以抽取真口袋；严格模式会跳过没有有效口袋的记录，不混入
    full-sequence fallback。

    伪肽链表示的说明（pocket 模式下适用）：口袋残基空间聚集但序列不连续，
    拼成伪肽不是声称它们共价相连，而是因为 (a) molBERT 只吃 FDDFS token；
    (b) 伪肽保留残基身份和相对顺序；(c) LOGOS 也做了口袋线性化。
    完整序列和口袋下标都存进 meta，换表示不用重解析。

    另注：本数据来自 PDBbind-cn 的某个快照，**不是** LOGOS 用的 v2016
    refined/core 划分，横向对比时要说明。
    """

    import numpy as np
    import pyarrow.parquet as pq
    from rdkit import Chem

    files = sorted(path.glob("*.parquet")) if path.is_dir() else [path]
    if not files:
        raise FileNotFoundError(f"{path} 下没有 parquet")

    n_bad = n_long = 0
    n_pocket = n_fullseq = 0
    n_coord_parse_error = 0
    n_receptor_invalid = 0
    n_ligand_no_finite_atom = 0
    n_ligand_atom_count_mismatch = 0
    n_invalid_target_smiles = 0
    n_ligand_masked_rows = 0
    n_ligand_atom_coord_rows = 0

    def bump(key: str, value: int = 1) -> None:
        if stats is not None:
            stats[key] = int(stats.get(key, 0)) + int(value)

    need = ["name", "seq", "smiles", "receptor_xyz", "ligand_xyz"]
    for fp in files:
        pf = pq.ParquetFile(fp)
        missing = set(need) - set(pf.schema_arrow.names)
        if missing:
            raise ValueError(f"{fp} 缺列 {sorted(missing)}，实得 {pf.schema_arrow.names}")

        # 按批流式读。整表 to_pylist() 会把 680 MB 的坐标数组一次性物化成
        # Python 对象，又慢又吃内存（实测跑不完）。
        for row in _iter_parquet_rows(pf, need, batch_size=256):
            bump("n_raw_rows")
            sid = f"pdbbind_{row.get('name','?')}"
            seq = (row.get("seq") or "").strip().upper()
            smi = (row.get("smiles") or "").strip()
            if not seq or not smi:
                n_bad += 1
                bump("n_bad_rows")
                continue
            target_mol = Chem.MolFromSmiles(smi)
            if target_mol is None:
                n_invalid_target_smiles += 1
                bump("n_invalid_target_smiles")
                continue
            expected_ligand_atoms = int(target_mol.GetNumAtoms())

            # 先试口袋模式。ligand_xyz 的 NaN 行是 SMILES 非原子 token 的
            # 合法占位，先筛掉它们，再用有限的原子坐标计算距离。
            pocket: str | None = None
            idx: Any = None
            ligand_atom_count = 0
            fallback_reason = "no_pocket_within_cutoff"
            try:
                rec = _as_xyz(row["receptor_xyz"], "receptor_xyz", sid)
                lig = _as_xyz(row["ligand_xyz"], "ligand_xyz", sid)
                idx, n_atom_rows, n_masked_rows, fallback_reason = _select_pocket_indices(
                    seq,
                    rec,
                    lig,
                    pocket_cutoff=pocket_cutoff,
                    min_pocket_residues=min_pocket_residues,
                )
                ligand_atom_count = n_atom_rows
                n_ligand_atom_coord_rows += n_atom_rows
                bump("n_ligand_atom_coord_rows", n_atom_rows)
                if n_masked_rows:
                    n_ligand_masked_rows += 1
                    bump("n_records_with_masked_ligand_tokens")
                    bump("n_masked_ligand_token_rows", n_masked_rows)

                if fallback_reason == "receptor_xyz_not_finite_or_sequence_length_mismatch":
                    n_receptor_invalid += 1
                    bump("n_receptor_coordinate_invalid")
                elif fallback_reason == "no_finite_ligand_atom_coords":
                    n_ligand_no_finite_atom += 1
                    bump("n_ligand_without_finite_atom_coords")
                if not fallback_reason and n_atom_rows != expected_ligand_atoms:
                    n_ligand_atom_count_mismatch += 1
                    bump("n_ligand_atom_coordinate_count_mismatch")
                    fallback_reason = "ligand_atom_coordinate_count_mismatch"
                if not fallback_reason:
                    pocket = "".join(seq[i] for i in idx)
            except (TypeError, ValueError):
                n_coord_parse_error += 1
                bump("n_coordinate_parse_error")
                fallback_reason = "coordinate_shape_parse_error"

            if pocket is not None:
                cond_seq, task, mode = pocket, "pocket2ligand", "pocket"
                n_pocket += 1
                bump("n_pocket")
            else:
                n_fullseq += 1
                bump("n_full_sequence")
                if require_pocket:
                    bump("n_without_pocket_skipped")
                    continue
                cond_seq, task, mode = seq, "protein2ligand", "full_sequence"

            if max_residues and len(cond_seq) > max_residues:
                n_long += 1
                bump("n_over_max_residues")
                continue
            bump("n_yielded")
            bump(f"n_{mode}_yielded")

            meta: dict[str, Any] = {
                "benchmark": benchmark,
                "condition_mode": mode,
                "n_condition_residues": len(cond_seq),
                "n_total_residues": len(seq),
                "pocket_source": "receptor_calpha_to_ligand_atom_xyz"
                if mode == "pocket" else None,
                "ligand_atom_coordinate_rows": int(ligand_atom_count),
                "ligand_rdkit_atom_count": int(expected_ligand_atoms),
            }
            if mode == "pocket":
                meta["pocket_cutoff"] = pocket_cutoff
                meta["pocket_indices"] = [int(i) for i in idx]
                meta["full_seq"] = seq
            else:
                meta["pocket_fallback_reason"] = fallback_reason

            yield RawSample(
                sample_id=sid,
                task=task,
                target_type="small_molecule",
                target_smiles=smi,
                condition_sequence=cond_seq,
                condition_type="protein",  # 条件是蛋白，target 才是小分子
                meta=meta,
            )

    print(
        f"    条件模式: 口袋 {n_pocket:,} / 全序列 {n_fullseq:,}"
        f"  (跳过: 数据异常 {n_bad:,}, 超 {max_residues} 残基 {n_long:,})",
        file=sys.stderr,
    )
    if n_fullseq > n_pocket and not require_pocket:
        print(
            "    !! 多数样本走了全序列模式（没有足够的可用 pocket 坐标），"
            "产出的是 protein2ligand 不是 pocket2ligand，报结果时注意区分",
            file=sys.stderr,
        )
    elif require_pocket and n_fullseq:
        print(
            f"    (严格 pocket 模式：跳过 {n_fullseq:,} 条没有有效 pocket 的记录)",
            file=sys.stderr,
        )
    if stats is not None:
        stats.update(
            {
                "n_raw_rows": int(stats.get("n_raw_rows", 0)),
                "n_bad_rows": int(n_bad),
                "n_over_max_residues": int(n_long),
                "n_pocket": int(n_pocket),
                "n_full_sequence": int(n_fullseq),
                "n_coordinate_parse_error": int(n_coord_parse_error),
                "n_receptor_coordinate_invalid": int(n_receptor_invalid),
                "n_ligand_without_finite_atom_coords": int(n_ligand_no_finite_atom),
                "n_ligand_atom_coordinate_count_mismatch": int(n_ligand_atom_count_mismatch),
                "n_invalid_target_smiles": int(n_invalid_target_smiles),
                "n_records_with_masked_ligand_tokens": int(n_ligand_masked_rows),
                "n_ligand_atom_coord_rows": int(n_ligand_atom_coord_rows),
            }
        )


READERS: dict[str, Callable[..., Iterator[RawSample]]] = {
    "unconditional": read_unconditional,
}


# ---------------------------------------------------------------------------
# 编码（多进程）
# ---------------------------------------------------------------------------

_GUIDE: CorpusGuide | None = None
_FAST: FastCorpusEncoder | None = None


def _init_worker(guide_path: str, use_fast: bool) -> None:
    global _GUIDE, _FAST
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 1_000_000))
    _GUIDE = load_corpus_guide(Path(guide_path))
    _FAST = FastCorpusEncoder(_GUIDE) if use_fast else None


def _encode_seq(sample_id: str, target_type: str, smiles: str | None, sequence: str | None) -> list[str]:
    """按当前 worker 的后端编码单个分子。"""

    if _FAST is not None:
        return encode_molecule_fast(
            target_type=target_type,
            sample_id=sample_id,
            encoder=_FAST,
            smiles=smiles,
            sequence=sequence,
        )
    assert _GUIDE is not None
    return encode_molecule(
        target_type=target_type,
        sample_id=sample_id,
        corpus_guide=_GUIDE,
        smiles=smiles,
        sequence=sequence,
    )


def _encode_one(s: RawSample) -> dict[str, Any]:
    assert _GUIDE is not None
    try:
        target_seq = None
        if s.target_smiles or s.target_sequence:
            target_seq = _encode_seq(
                f"{s.sample_id}#tgt", s.target_type, s.target_smiles, s.target_sequence
            )
        cond_seq = None
        if s.condition_smiles or s.condition_sequence:
            # 用 condition_type，不是 target_type —— 跨模态任务两者不同
            cond_seq = _encode_seq(
                f"{s.sample_id}#cond",
                s.condition_type or s.target_type,
                s.condition_smiles,
                s.condition_sequence,
            )
        # 纯文本条件的评测样本（如 MolCustom）没有 target 也没有条件分子，
        # 这是合法的——生成阶段才产出分子。只有三者全空才算异常。
        if target_seq is None and cond_seq is None and not s.condition_text:
            return {"_error": "既无 target 也无 condition 也无文本", "_id": s.sample_id}
        return {
            "record": make_record(
                sample_id=s.sample_id,
                task=s.task,
                target_type=s.target_type,
                target_fddfs=target_seq or [],
                condition_text=s.condition_text,
                condition_molecule_fddfs=cond_seq,
                condition_modality=s.condition_type,
                condition_role=s.condition_role,
                condition_instruction=s.condition_instruction,
                meta=s.meta,
            )
        }
    except UnknownFeatureError as exc:
        # 语料 vocab 未覆盖的原子/键类型。与纯 Python 路径同义：整条按未知 token 丢弃，
        # 不记成 error（保持与 batch12 的 drop_rate 口径一致）。
        return {"_unknown": sorted(exc.tokens), "_id": s.sample_id}
    except Exception as exc:  # noqa: BLE001 — 单条失败不中断整批
        return {"_error": f"{type(exc).__name__}: {exc}", "_id": s.sample_id}


def _encode_chunk(chunk: list[RawSample]) -> list[dict[str, Any]]:
    """按 chunk 编码，摊薄单条 submit 的 IPC 开销。"""

    return [_encode_one(s) for s in chunk]


def _chunked(it: Iterable[RawSample], size: int) -> Iterator[list[RawSample]]:
    """把惰性样本流切成定长 chunk，全程不物化整体。"""

    iterator = iter(it)
    while True:
        block = list(itertools.islice(iterator, size))
        if not block:
            return
        yield block


def _imap_bounded(ex: ProcessPoolExecutor, chunks: Iterator[list[RawSample]], max_inflight: int) -> Iterator[dict[str, Any]]:
    """有界 in-flight 的并行 map。

    原来用 ``ex.map(_encode_one, samples, chunksize=64)``，它会把**全部** chunk
    一次性 pickle 进 ``_pending_work_items``（等于父进程里又一份完整拷贝），
    结果也堆在 Future 里等消费。蛋白一条 8000+ 个字符串 token，这样必然 OOM。
    这里改成滑动窗口：同时在飞的 chunk 数不超过 ``max_inflight``。

    产出顺序不保证与输入一致（每条 record 自带 ``id``，训练前本来也要 shuffle）。
    """

    pending: set[Any] = set()
    for chunk in chunks:
        pending.add(ex.submit(_encode_chunk, chunk))
        if len(pending) >= max_inflight:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for fut in done:
                yield from fut.result()
    while pending:
        done, pending = wait(pending, return_when=FIRST_COMPLETED)
        for fut in done:
            yield from fut.result()


def _imap_bounded_ordered(
    ex: ProcessPoolExecutor,
    chunks: Iterator[list[RawSample]],
    max_inflight: int,
) -> Iterator[dict[str, Any]]:
    """Bound pending work while publishing encoded chunks in source order."""

    pending: deque[Any] = deque()
    exhausted = False
    while pending or not exhausted:
        while not exhausted and len(pending) < max_inflight:
            try:
                pending.append(ex.submit(_encode_chunk, next(chunks)))
            except StopIteration:
                exhausted = True
        if pending:
            yield from pending.popleft().result()


def run(
    samples: Iterable[RawSample],
    out_path: Path,
    guide_path: Path,
    workers: int,
    dataset_name: str,
    *,
    use_fast: bool = True,
    chunk_size: int = 32,
    inflight_per_worker: int = 4,
    preserve_order: bool = False,
) -> dict[str, Any]:
    """并行编码 → 流式写入 → 抽样验收。

    ``samples`` 是**惰性可迭代对象**，不要传 list —— 全量物化正是 batch3 OOM 的
    第一重放大（524,013 条蛋白 RawSample 攒在父进程，再 fork×62）。

    改动（2026-08-17）：
    - 输入流式化，父进程不再持有全部样本；
    - ``ex.map`` 换成有界 in-flight 的滑动窗口（见 ``_imap_bounded``）；
    - 编码默认走 C++ 热区（``use_fast``），纯 Python 版对蛋白是 O(N²)+递归爆栈。
    """

    started = time.time()
    n_in = n_ok = n_unknown = 0
    errors: list[str] = []
    unknown_tokens: Counter[str] = Counter()
    seqs_for_validation: list[list[str]] = []  # 只收集前 1000 条用于验收

    max_inflight = max(2, int(workers) * int(inflight_per_worker))
    print(f"    编码后端: {'C++ FastFDDFS' if use_fast else 'Python(慢，仅回退用)'}  "
          f"{workers} 进程  chunk={chunk_size}  in-flight<={max_inflight} chunk",
          flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    guide = load_corpus_guide(guide_path)
    vocab_tokens = base_vocab_tokens(guide)

    with out_path.open("w", encoding="utf-8") as f, ProcessPoolExecutor(
        max_workers=workers, initializer=_init_worker, initargs=(str(guide_path), bool(use_fast))
    ) as ex:
        chunks = _chunked(samples, chunk_size)
        encoded = (
            _imap_bounded_ordered(ex, chunks, max_inflight)
            if preserve_order
            else _imap_bounded(ex, chunks, max_inflight)
        )
        for res in encoded:
            n_in += 1
            if "_error" in res:
                errors.append(f"{res['_id']}: {res['_error']}")
                continue
            if "_unknown" in res:
                n_unknown += 1
                unknown_tokens.update(res["_unknown"])
                continue

            rec = res["record"]
            # 纯 Python 后端不做 feature 预检，这里补一次未知 token 过滤
            bad_tokens: set[str] = set()
            bad_tokens |= unknown_tokens_in(rec.get("target_fddfs") or [], vocab_tokens)
            cond_mol = (rec.get("condition") or {}).get("molecule_fddfs") or []
            bad_tokens |= unknown_tokens_in(cond_mol, vocab_tokens)
            if bad_tokens:
                n_unknown += 1
                unknown_tokens.update(bad_tokens)
                continue

            # 通过过滤，写盘
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_ok += 1

            # 只收集前 1000 条用于验收（避免大数据集内存爆，蛋白序列尤其长）
            if len(seqs_for_validation) < 1000:
                if rec.get("target_fddfs"):
                    seqs_for_validation.append(rec["target_fddfs"])
                if cond_mol:
                    seqs_for_validation.append(cond_mol)

            if n_ok % 20000 == 0:
                print(f"      已写出 {n_ok:,}（读入 {n_in:,}）  "
                      f"{n_in/max(1e-9,time.time()-started):,.0f} 条/秒", flush=True)

    elapsed = time.time() - started
    drop_rate = n_unknown / n_in if n_in else 0.0
    print(f"    完成 {n_ok:,} / 失败 {len(errors):,} / 丢弃未知token {n_unknown:,} "
          f"({drop_rate:.2%}), {elapsed:.0f}s", flush=True)
    if errors:
        print(f"    失败样例: {errors[:3]}", file=sys.stderr)
    if n_ok == 0:
        out_path.unlink(missing_ok=True)
        raise ValidationError(
            f"{dataset_name}: 一条都没通过（读入 {n_in:,}，失败 {len(errors):,}，"
            f"未知 token 丢弃 {n_unknown:,}）。失败样例: {errors[:3]}"
        )
    if drop_rate > 0.20:
        out_path.unlink(missing_ok=True)
        raise ValidationError(
            f"{dataset_name}: 未知 token 丢弃率 {drop_rate:.1%} 超过 20% 阈值，"
            "这不像分布差异而像系统性错误"
        )

    # 验收（抽样，内存占用小）
    if seqs_for_validation:
        report = validate_encoded(seqs_for_validation, guide, dataset_name=dataset_name)
        print(f"    ✓ 验收通过 (抽样 {len(seqs_for_validation):,} 条序列, "
              f"R token {report['n_ring_tokens']:,})", flush=True)
    else:
        report = {"note": "纯文本条件评测集，无分子序列可验收"}
        print("    ✓ 纯文本条件评测集，跳过分子序列验收", flush=True)

    print(f"    写出 {out_path.name}: {n_ok:,} 行", flush=True)

    return {
        "output": str(out_path),
        "serializer_backend": "cpp_fast_fddfs" if use_fast else "python_fddfs",
        "workers": int(workers),
        "n_input": n_in,
        "n_ok": n_ok,
        "n_error": len(errors),
        "n_dropped_unknown_token": n_unknown,
        "preserve_input_order": bool(preserve_order),
        "drop_rate": round(drop_rate, 5),
        "distinct_unknown_tokens": len(unknown_tokens),
        "top_unknown_tokens": [t for t, _ in unknown_tokens.most_common(10)],
        "error_samples": errors[:10],
        "elapsed_sec": round(elapsed, 1),
        "validation": report,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=sorted(READERS))
    ap.add_argument("--input", type=Path, required=True, help="输入文件或目录")
    ap.add_argument("--output", type=Path, required=True, help="输出 jsonl")
    ap.add_argument(
        "--benchmark",
        choices=ACTIVE_UNCONDITIONAL_BENCHMARKS,
        default=None,
        help="正式无条件 benchmark 名（unconditional 必填）",
    )
    ap.add_argument("--guide", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=0, help=">0 时只取前 N 条，用于冒烟（在读取阶段生效）")
    # 默认不再是 cpu_count-2：蛋白单条图很重（1024 残基≈8000 重原子），
    # 进程数过高时每个 worker 的常驻内存（guide + 图 + token 列表）会把节点撑爆。
    ap.add_argument("--workers", type=int, default=max(1, min(16, (os.cpu_count() or 8) - 2)))
    ap.add_argument("--chunk-size", type=int, default=32, help="每次 submit 的样本数")
    ap.add_argument("--inflight-per-worker", type=int, default=4, help="每个 worker 允许在飞的 chunk 数")
    ap.add_argument("--python-serializer", action="store_true",
                    help="强制用纯 Python FDDFS（O(N²)+递归，仅小分子可用；默认走 C++ 热区）")
    args = ap.parse_args()

    guide_path = args.guide or DEFAULT_GUIDE_PATH
    g = load_corpus_guide(guide_path)
    print(f"guide: {guide_path}")
    print(f"  triplet={g.triplet_size:,} vocab={len(g.vocab['token_to_id']):,} "
          f"ring_base={g.vocab['ring_base']}")

    reader = READERS[args.dataset]
    kwargs: dict[str, Any] = {}
    reader_stats: dict[str, int] = {}
    if args.dataset == "unconditional":
        if not args.benchmark:
            ap.error("--dataset unconditional 需要 --benchmark")
        kwargs["benchmark"] = args.benchmark

    print(f"\n=== {args.dataset} ← {args.input} ===")
    # 流式：reader 本来就是惰性生成器，绝不要 list() 它。
    # `--limit` 也在读取阶段用 islice 生效 —— 原来是 list() 之后再切片，
    # 等于冒烟测试照样先物化全量（batch3 那几次 "100sample/1k" 日志就是这么假的）。
    samples: Iterator[RawSample] = iter(reader(args.input, **kwargs))
    if args.limit > 0:
        samples = itertools.islice(samples, args.limit)
        print(f"    --limit 生效，只取前 {args.limit:,} 条（读取阶段截断）")

    try:
        summary = run(
            samples,
            args.output,
            guide_path,
            args.workers,
            args.dataset,
            use_fast=not args.python_serializer,
            chunk_size=args.chunk_size,
            inflight_per_worker=args.inflight_per_worker,
        )
    except ValidationError as exc:
        print(f"\n!! 验收失败，不产出数据:\n{exc}", file=sys.stderr)
        return 1

    meta_path = args.output.with_suffix(".summary.json")
    if reader_stats:
        summary["reader_stats"] = reader_stats
    meta_path.write_text(
        json.dumps({"dataset": args.dataset, "input": str(args.input),
                    "guide": str(guide_path), **summary},
                   ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\n{args.dataset.upper()}_PREPROCESS_OK")
    return 0
