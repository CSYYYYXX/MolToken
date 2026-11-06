"""统一的 FDDFS 编码入口：任意分子表示 → FDDFS token 序列。

**所有**下游任务数据集的预处理都必须走这个模块，不许各写各的。

背景（2026-08-14 踩的坑）：第一版 USPTO-50K 预处理直接存了 SMILES 字符串；
改成走 pipeline 后又用错了序列化算法（stackK 而非 FDDFS），且 frequency guide
是拿那 8 万个分子现场建的、不是语料级的。两个错误都要重做 4 万样本。

三条必须遵守的规则：

1. **算法用 FDDFS**，不是 stackK。预训练语料 ``count_summary.json`` 写着
   ``serializer: "fddfs"``、``ring_base: 334``、``k_base: null``；vocab 里
   334..582 那段是 ring id。stackK 产 ``K<num>``，模型没见过。
2. **guide 用语料级的**，从 ``fddfs_guide_recount_v1/guide_counts.json`` 加载。
   guide 决定 DFS 遍历顺序，``transition_key_from_features`` 的排序主键是
   ``triplet_counts``；用局部数据现建的 guide 会给出另一种遍历顺序，
   等于喂给模型另一种方言。
3. **落盘前跑 ``validate_encoded()``**，任一项不过就别产出。
"""

from __future__ import annotations

from pipeline import paths as _moltoken_paths

import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from pipeline.graph.chem_graph_builder import (
    is_standard_nucleotide_sequence,
    is_standard_protein_sequence,
    nucleotide_sequence_to_graph_data_fast,
    protein_sequence_to_graph_data_fast,
    sequence_to_peptide_graph_data,
    smiles_to_graph_data,
)
from pipeline.serialization.fddfs import (
    build_vocab,
    deserialize_fddfs,
    is_fddfs_ring_token,
    serialize_fddfs,
)
from pipeline.serialization.freq_dfs_core import FrequencyGuide, graph_signature
from pipeline.finetune.frozen_axis_errors import (
    RingIdOverflow,
    UnknownFeatureError,
)

DATA_ROOT = Path(
    str(_moltoken_paths.LOCAL_DATA_ROOT / '')
)
#: Phase 0.2 重跑 count 的产物。旧的 ``fddfs_heavy_atom_normv2_compact_v1/
#: guide_counts.json`` **不能用**——它只存了 triplet_unique 计数。
DEFAULT_GUIDE_PATH = _moltoken_paths.TOKENIZER_ASSET_ROOT / "guide_counts.json"

TARGET_TYPES: tuple[str, ...] = (
    "small_molecule",
    "peptide",
    "protein",
    "dna",
    "rna",
)


@dataclass(frozen=True)
class CorpusGuide:
    """语料级 frequency guide + 序列化所需的附属信息。"""

    guide: FrequencyGuide
    default_edge_feature: tuple[int, ...]
    vocab: dict[str, Any]
    source_path: Path

    @property
    def triplet_size(self) -> int:
        return len(self.guide.triplet_counts)


def load_corpus_guide(path: Path | None = None) -> CorpusGuide:
    """加载语料级 guide。

    走 ``serialize_fddfs_full.load_count_payload``，
    它会在遇到旧格式（没有 triplet_counts）时显式报错而不是静默降级。
    """

    path = Path(path) if path is not None else DEFAULT_GUIDE_PATH
    if not path.is_file():
        raise FileNotFoundError(
            f"找不到语料级 guide: {path}\n"
            "Provide a full frequency guide or install the bundled MolToken assets."
        )

    from cli.pretrain.serialize_fddfs_full import build_guide, load_count_payload

    payload = load_count_payload(path)
    guide = build_guide(payload)
    default_edge = payload["default_edge_feature"]
    vocab = build_vocab(
        payload["node_counts"],
        payload["edge_counts"],
        default_edge,
        omit_default_edge=True,
    )
    return CorpusGuide(
        guide=guide,
        default_edge_feature=default_edge,
        vocab=vocab,
        source_path=path,
    )


def build_graph(
    *,
    target_type: str,
    sample_id: str,
    smiles: str | None = None,
    sequence: str | None = None,
) -> dict[str, Any]:
    """按模态分派到对应的 heavy atom graph 构建器。

    小分子走 SMILES，其余四种模态走序列。蛋白/多肽和核酸都优先用模板快速路径
    （只覆盖标准残基/碱基），非标准字符回退到 RDKit 全分子构建。
    """

    if target_type not in TARGET_TYPES:
        raise ValueError(f"未知 target_type={target_type!r}，应为 {TARGET_TYPES}")

    if target_type == "small_molecule":
        if not smiles:
            raise ValueError(f"{sample_id}: small_molecule 需要 smiles")
        return smiles_to_graph_data(smiles, sample_id=sample_id)

    if not sequence:
        raise ValueError(f"{sample_id}: {target_type} 需要 sequence")
    seq = sequence.strip().upper()

    if target_type in ("protein", "peptide"):
        if is_standard_protein_sequence(seq):
            return protein_sequence_to_graph_data_fast(seq, sample_id=sample_id)
        # 含非标准残基，回退 RDKit（慢，且可能失败）
        return sequence_to_peptide_graph_data(seq, sample_id=sample_id)

    # dna / rna
    if not is_standard_nucleotide_sequence(seq, target_type):
        raise ValueError(f"{sample_id}: {target_type} 含非标准碱基，无法构图")
    return nucleotide_sequence_to_graph_data_fast(seq, target_type, sample_id=sample_id)


#: 语料为 ring id 预留的槽位数。推导（全部实测，2026-08-17 复核）：
#:   fddfs_heavy_atom_normv2_compact_v1/count_summary.json : base_vocab_size=334, ring_base=334
#:   codebook_fddfs_prefix70_mf80000_v1/summary.json       : max_new_id=17203, total_merges=16621
#:   => 首个 BPE merge id = 17203 - 16621 + 1 = 583
#:   => ring 区 = 334..582，共 249 个（R1..R249）
#: 超过这个数的 ring id 会被映射到 >=583，**静默撞进 BPE merges 区**，
#: 而 encode_fddfs_token / validate_encoded 都不查上界。这里显式兜住。
CORPUS_RING_CAPACITY = 249


class FastCorpusEncoder:
    """C++ FDDFS 编码器 + int id → string token 反查。

    **为什么需要它**：纯 Python 的 ``serialize_fddfs`` 有两个致命问题，对小分子
    无所谓、对蛋白是灾难性的（2026-08-17 定位）：

    1. ``freq_dfs_core.FrequencyGuide.root_key`` 内部调 ``node_features(graph)``
       重建全图 feature，而 ``make_dfs_plan`` 对每个节点各调一次 → **O(N²)**。
       蛋白 N≈8000 时单条约 2 分钟；小分子 N≈30 时可忽略。
       这就是 batch12 通过、batch3 全灭的真正原因。
    2. ``dfs`` / ``emit_subtree`` 是递归的，深度 O(N)，蛋白必然 RecursionError。

    C++ 版两个问题都没有：``root_key`` 接收已算好的 nfeat（O(1)），遍历用显式栈。
    全量语料（3.23 亿条）本来就是这个 C++ 路径跑的，这里只是让下游数据集也用上。

    **输出格式**：返回 **字符串 token 列表**，与纯 Python 路径逐 token 一致，
    因此与 batch12 已落盘的 8 个数据集格式兼容（``target_fddfs`` 是字符串数组）。
    """

    def __init__(self, corpus_guide: CorpusGuide, *, ring_capacity: int = CORPUS_RING_CAPACITY) -> None:
        from pipeline.serialization.stackk_fast import (
            FAST_FDDFS_AVAILABLE,
            FAST_FDDFS_ERROR,
            FastFDDFSEncoder,
        )

        if not FAST_FDDFS_AVAILABLE:
            raise ImportError(
                "C++ FDDFS 后端不可用，无法走快速路径。\n"
                f"  import_error={FAST_FDDFS_ERROR}\n"
                "  期望的扩展: pipeline/serialization/_stackk_cpp*.so（cpython-311）"
            )

        vocab = corpus_guide.vocab
        self._encoder = FastFDDFSEncoder(corpus_guide.guide, vocab)
        self.ring_base = int(vocab["ring_base"])
        self.ring_min = int(vocab.get("ring_min", 1))
        self.ring_capacity = int(ring_capacity)
        self.default_edge_feature = tuple(int(x) for x in corpus_guide.default_edge_feature)
        # id -> token 反查表。base vocab 是 0..ring_base-1；>=ring_base 一律是 ring。
        self._id_to_token: dict[int, str] = {
            int(i): t for t, i in vocab["token_to_id"].items()
        }
        self._vocab_tokens: set[str] = set(vocab["token_to_id"])
        self.max_ring_id_seen = 0

    def check_features(self, graph: dict[str, Any]) -> None:
        """编码前先查 feature 覆盖，未覆盖的抛 ``UnknownFeatureError``。

        C++ 遇到不在 vocab 的 feature 会直接 throw，且错误信息里不带是哪个 feature。
        纯 Python 路径则是照常发出 token、由 ``unknown_tokens_in`` 过滤掉整条样本。
        为了和 batch12 的口径保持一致（summary 里要报 ``top_unknown_tokens``），
        这里在调 C++ 之前先做一次 O(N) 预检，把具体 token 名字捞出来。
        """

        import numpy as np

        bad: set[str] = set()
        for row in np.asarray(graph["node_features"], dtype=np.int64).reshape((-1, 6)):
            tok = "N:" + ",".join(str(int(x)) for x in row)
            if tok not in self._vocab_tokens:
                bad.add(tok)
        for row in np.asarray(graph["edge_features"], dtype=np.int64).reshape((-1, 3)):
            feat = tuple(int(x) for x in row)
            # 默认边被 omit_default_edge 有意排除在 vocab 之外，不算未知
            if feat == self.default_edge_feature:
                continue
            tok = "E:" + ",".join(str(x) for x in feat)
            if tok not in self._vocab_tokens:
                bad.add(tok)
        if bad:
            raise UnknownFeatureError(bad)

    def ids_to_tokens(self, ids: Sequence[int]) -> list[str]:
        """int token id → 字符串 token。>=ring_base 的一律还原成 ``R<num>``。"""

        out: list[str] = []
        for raw in ids:
            i = int(raw)
            tok = self._id_to_token.get(i)
            out.append(tok if tok is not None else f"R{i - self.ring_base + self.ring_min}")
        return out

    def serialize(self, graph: dict[str, Any], sample_id: str) -> list[str]:
        """graph → FDDFS 字符串 token 序列（走 C++ 热区）。"""

        self.check_features(graph)
        res = self._encoder.serialize_ids(graph)
        max_ring = int(res["stats"].get("max_ring_id", 0))
        if max_ring > self.ring_capacity:
            raise RingIdOverflow(
                sample_id=sample_id,
                max_ring_id=max_ring,
                ring_capacity=self.ring_capacity,
                first_colliding_token_id=self.ring_base + self.ring_capacity,
            )
        if max_ring > self.max_ring_id_seen:
            self.max_ring_id_seen = max_ring
        return self.ids_to_tokens(res["ids"])


def encode_molecule_fast(
    *,
    target_type: str,
    sample_id: str,
    encoder: FastCorpusEncoder,
    smiles: str | None = None,
    sequence: str | None = None,
) -> list[str]:
    """任意分子表示 → FDDFS token 序列（C++ 热区版）。

    与 ``encode_molecule`` 输出逐 token 一致，但不带 ``verify`` 选项 ——
    roundtrip 校验走的 ``SequenceParser`` 也是递归实现，蛋白上同样会爆栈，
    抽检请用 ``validation/verify_fddfs_roundtrip_samples.py``（它设了 recursionlimit）。
    """

    graph = build_graph(
        target_type=target_type,
        sample_id=sample_id,
        smiles=smiles,
        sequence=sequence,
    )
    return encoder.serialize(graph, sample_id)


def encode_molecule(
    *,
    target_type: str,
    sample_id: str,
    corpus_guide: CorpusGuide,
    smiles: str | None = None,
    sequence: str | None = None,
    verify: bool = False,
) -> list[str]:
    """任意分子表示 → FDDFS token 序列（完整 pipeline）。

    ``verify=True`` 时顺带做一次 roundtrip 拓扑比对，慢一倍，用于抽检。
    """

    graph = build_graph(
        target_type=target_type,
        sample_id=sample_id,
        smiles=smiles,
        sequence=sequence,
    )
    result = serialize_fddfs(graph, corpus_guide.guide, omit_default_edge=True)

    if verify:
        decoded = deserialize_fddfs(result.sequence, corpus_guide.default_edge_feature)
        if graph_signature(graph, result.old_to_new) != graph_signature(decoded):
            raise ValueError(f"{sample_id}: FDDFS roundtrip 拓扑不一致")

    return result.sequence


# ---------------------------------------------------------------------------
# 验收断言（Phase 0.4）
#
# 上一轮 4 万样本全用错算法却以为成功，就是因为一个检查都没有。
# 落盘前必须过这一关，任一项失败直接抛错，不产出半成品。
# ---------------------------------------------------------------------------


CONDITION_TYPES: tuple[str, ...] = ("none", "text", "molecule", "both")


def condition_metadata_for_task(
    task: str,
    target_type: str,
    condition_modality: str | None,
    *,
    has_molecule: bool,
    has_text: bool,
) -> tuple[str, str | None, str | None]:
    """Return stable, task-agnostic metadata for a condition stream.

    ``condition.type`` is the transport shape (text/molecule/both).  The
    modality and role fields describe what the source means without adding a
    task-specific token to the Qwen vocabulary.  Instructions are ordinary
    text consumed by the existing MPNet encoder, so new tasks can introduce a
    description without resizing the language model.
    """

    task = str(task)
    if has_molecule:
        # Older materialized records used ``molecule`` as the transport kind,
        # not as a true source modality.  Treat it as unspecified here.
        supplied_modality = None if condition_modality in {None, "molecule"} else condition_modality
        # The target type is not necessarily the source type: PDBBind and
        # full-protein ligand records target a small molecule while their
        # condition is a protein pocket/sequence.  Keep that distinction in
        # the extensible metadata instead of silently labeling the protein as
        # a small molecule.
        if supplied_modality:
            modality = str(supplied_modality)
        elif task in {"pocket2ligand", "protein2ligand"}:
            modality = "protein"
        else:
            modality = str(target_type)
    elif has_text:
        modality = "text"
    else:
        modality = "none"
    if modality not in TARGET_TYPES and modality not in {"text", "none"}:
        raise ValueError(f"unsupported condition modality: {modality!r}")

    role: str | None = None
    instruction: str | None = None
    if task == "forward_synthesis":
        role = "reactants_and_reagents"
        instruction = "Generate the product molecule from the given reactants and reagents."
    elif task == "retrosynthesis":
        role = "product"
        instruction = "Predict the reactants and reagents from the given product molecule."
    elif task == "pocket2ligand":
        role = "protein_pocket"
        instruction = "Generate a small molecule ligand for the given protein pocket sequence."
    elif task == "protein2ligand":
        role = "protein_sequence"
        instruction = "Generate a small molecule ligand for the given protein sequence."
    elif task == "text2mol":
        role = "molecule_description"
        instruction = "Generate a small molecule matching the given description."
    elif task == "text2protein":
        role = "protein_description"
        instruction = "Generate a protein sequence matching the given description."
    elif task.startswith("moledit_"):
        role = "source_molecule_and_edit_instruction"
        instruction = "Apply the requested edit to the input molecule."
    elif task.startswith("molopt_"):
        role = "source_molecule_and_optimization_instruction"
        instruction = "Optimize the input molecule according to the requested objective."
    elif task.startswith("molcustom_"):
        role = "molecule_constraint"
        instruction = "Generate a molecule satisfying the requested structural constraint."
    elif has_molecule:
        role = "structural_condition"
        instruction = "Generate the target molecule from the given structural condition."
    elif has_text:
        role = "text_condition"
        instruction = "Generate the target from the given description."

    return modality, role, instruction


def make_record(
    *,
    sample_id: str,
    task: str,
    target_type: str,
    target_fddfs: Sequence[str],
    condition_text: str | None = None,
    condition_molecule_fddfs: Sequence[str] | None = None,
    condition_modality: str | None = None,
    condition_role: str | None = None,
    condition_instruction: str | None = None,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """构造统一的微调样本 record。**所有**数据集共用这个 schema。

    ``condition.type`` 由传入的字段自动推断，对应四种输入形态：
    无条件 / 纯文本 / 纯分子 / 文本+分子。训练时按它决定拼哪几段 embedding。
    """

    if target_type not in TARGET_TYPES:
        raise ValueError(f"未知 target_type={target_type!r}")

    explicit_has_text = bool((condition_text or "").strip()) or bool((condition_instruction or "").strip())
    has_mol = condition_molecule_fddfs is not None
    inferred_modality, inferred_role, inferred_instruction = condition_metadata_for_task(
        task,
        target_type,
        condition_modality,
        has_molecule=has_mol,
        has_text=explicit_has_text,
    )
    if not explicit_has_text and not has_mol:
        # An unconditional record has no semantic source modality or role.
        # Canonicalize legacy callers that supplied the target modality through
        # ``RawSample.condition_type``; do not let metadata synthesize a text
        # prefix for a transport shape that is explicitly ``none``.
        modality = "none"
        role = None
        instruction = None
        has_text = False
    else:
        modality = condition_modality or inferred_modality
        role = condition_role or inferred_role
        instruction = condition_instruction or inferred_instruction
        has_text = explicit_has_text or bool((instruction or "").strip())
        if not has_text:
            instruction = None
        if has_text and not has_mol:
            modality = "text"
    cond_type = {
        (False, False): "none",
        (True, False): "text",
        (False, True): "molecule",
        (True, True): "both",
    }[(has_text, has_mol)]
    if cond_type == "none" and (
        modality != "none" or role is not None or instruction is not None
    ):
        raise AssertionError("unconditional condition metadata was not canonicalized")

    return {
        "id": sample_id,
        "task": task,
        "target_type": target_type,
        "condition": {
            "type": cond_type,
            "text": condition_text,
            "molecule_fddfs": list(condition_molecule_fddfs) if has_mol else None,
            "modality": modality,
            "role": role,
            "instruction": instruction,
        },
        "target_fddfs": list(target_fddfs),
        "meta": meta or {},
    }


class ValidationError(RuntimeError):
    """预处理验收未通过。"""


def base_vocab_tokens(corpus_guide: CorpusGuide) -> set[str]:
    """语料 base vocab 里的 node/edge token 集合。"""

    v = corpus_guide.vocab.get("token_to_id", corpus_guide.vocab)
    return set(v) if isinstance(v, dict) else set()


def unknown_tokens_in(seq: Sequence[str], vocab: set[str]) -> set[str]:
    """序列里不在语料 base vocab 的 node/edge token。

    括号和 ``R<num>`` 环 id 不查——前者恒在 vocab，后者是动态映射
    （``ring_base + num - 1``），不在 token_to_id 里。
    """

    out: set[str] = set()
    for t in seq:
        if t.startswith(("N:", "E:")) and t not in vocab:
            out.add(t)
    return out


def filter_unknown_token_records(
    records: list[dict[str, Any]],
    corpus_guide: CorpusGuide,
    *,
    dataset_name: str,
    max_drop_rate: float = 0.20,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """丢掉含未知 token 的样本，返回 (保留的, 报告)。

    为什么是丢弃而不是整批失败：下游数据集难免出现语料没覆盖的原子/电荷态
    （实测 ChEBI-20 有 Na+/Cl-/Fe/Ni2+，QM9 有 C+），**模型对这些 token 根本
    没有 embedding**，留着也没法用。少量丢弃是正常的数据分布差异。

    但比例过高（默认 >20%）就不是分布差异而是系统性错误了（例如序列化算法用错、
    guide 不匹配），这时抛 ``ValidationError``。
    """

    vocab = base_vocab_tokens(corpus_guide)
    kept: list[dict[str, Any]] = []
    dropped: list[str] = []
    all_unknown: Counter[str] = Counter()

    for rec in records:
        bad: set[str] = set()
        bad |= unknown_tokens_in(rec.get("target_fddfs") or [], vocab)
        cond = (rec.get("condition") or {}).get("molecule_fddfs") or []
        bad |= unknown_tokens_in(cond, vocab)
        if bad:
            all_unknown.update(bad)
            dropped.append(rec.get("id", "?"))
        else:
            kept.append(rec)

    n = len(records)
    rate = len(dropped) / n if n else 0.0
    report = {
        "n_before": n,
        "n_kept": len(kept),
        "n_dropped_unknown_token": len(dropped),
        "drop_rate": round(rate, 5),
        "distinct_unknown_tokens": len(all_unknown),
        "top_unknown_tokens": [t for t, _ in all_unknown.most_common(10)],
    }
    if rate > max_drop_rate:
        raise ValidationError(
            f"[{dataset_name}] 未知 token 丢弃率 {rate:.1%} 超过阈值 {max_drop_rate:.0%}，"
            f"这不像分布差异而像系统性错误。涉及 {len(all_unknown)} 种 token，"
            f"样例 {report['top_unknown_tokens'][:5]}"
        )
    return kept, report


def validate_encoded(
    sequences: Sequence[Sequence[str]],
    corpus_guide: CorpusGuide,
    *,
    dataset_name: str,
    roundtrip_graphs: Iterable[tuple[str, dict[str, Any]]] | None = None,
    roundtrip_n: int = 100,
    seed: int = 42,
) -> dict[str, Any]:
    """对一批已编码序列做验收。不通过抛 ``ValidationError``。

    检查四项：
      1. 出现过 ``R<num>``（说明确实是 FDDFS）
      2. 没有 ``K<num>``（说明不是 stackK）
      3. 所有 node/edge token 都在语料 base vocab 内
      4. 抽 ``roundtrip_n`` 条做拓扑 roundtrip（需传入原始 graph）
    """

    if not sequences:
        raise ValidationError(f"{dataset_name}: 没有任何序列")

    n_ring = n_stackk = 0
    unknown: set[str] = set()
    vocab_tokens = corpus_guide.vocab.get("token_to_id", corpus_guide.vocab)

    for seq in sequences:
        for tok in seq:
            if is_fddfs_ring_token(tok):
                n_ring += 1
            elif tok.startswith("K") and tok[1:].isdigit():
                n_stackk += 1
            elif tok in ("(", ")"):
                continue
            elif tok.startswith(("N:", "E:")):
                if isinstance(vocab_tokens, dict) and tok not in vocab_tokens:
                    unknown.add(tok)

    problems: list[str] = []
    if n_stackk:
        problems.append(
            f"出现 {n_stackk} 个 K<num> token —— 用的是 stackK 不是 FDDFS。"
            "检查 import 的是不是 pipeline.serialization.fddfs"
        )
    if n_ring == 0:
        problems.append(
            "一个 R<num> 都没有。要么全是无环分子（极不可能），"
            "要么序列化算法不对"
        )
    if unknown:
        sample = sorted(unknown)[:5]
        problems.append(
            f"{len(unknown)} 个 token 不在语料 base vocab 里，样例 {sample}。"
            "说明这批数据含语料没覆盖的原子/键类型"
        )

    report: dict[str, Any] = {
        "dataset": dataset_name,
        "n_sequences": len(sequences),
        "n_ring_tokens": n_ring,
        "n_stackk_tokens": n_stackk,
        "n_unknown_tokens": len(unknown),
        "guide_source": str(corpus_guide.source_path),
        "guide_triplet_size": corpus_guide.triplet_size,
    }

    if roundtrip_graphs is not None:
        pairs = list(roundtrip_graphs)
        rng = random.Random(seed)
        picked = rng.sample(pairs, min(roundtrip_n, len(pairs)))
        failed: list[str] = []
        for sample_id, graph in picked:
            try:
                res = serialize_fddfs(graph, corpus_guide.guide, omit_default_edge=True)
                dec = deserialize_fddfs(res.sequence, corpus_guide.default_edge_feature)
                if graph_signature(graph, res.old_to_new) != graph_signature(dec):
                    failed.append(sample_id)
            except Exception as exc:  # noqa: BLE001 — 记录后统一报
                failed.append(f"{sample_id}({type(exc).__name__})")
        report["roundtrip_checked"] = len(picked)
        report["roundtrip_failed"] = len(failed)
        if failed:
            problems.append(
                f"roundtrip 抽检 {len(picked)} 条失败 {len(failed)} 条，"
                f"样例 {failed[:5]}"
            )

    report["problems"] = problems
    if problems:
        raise ValidationError(
            f"[{dataset_name}] 验收未通过:\n  - " + "\n  - ".join(problems)
        )
    return report
