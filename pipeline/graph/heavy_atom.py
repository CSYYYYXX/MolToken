"""Heavy atom graph 构建和读取入口。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

from pipeline.paths import PipelinePaths


def graph_from_smiles(smiles: str, sample_id: str, properties: dict[str, Any] | None = None) -> dict[str, Any]:
    """把 SMILES 转成当前 heavy atom graph 字典格式。"""

    from pipeline.graph.chem_graph_builder import smiles_to_graph_data

    return smiles_to_graph_data(smiles, sample_id=sample_id, properties=properties)


def graph_from_sequence(
    sequence: str,
    sample_id: str,
    properties: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """把标准氨基酸 FASTA 序列转成当前 heavy atom graph 字典格式。"""

    from pipeline.graph.chem_graph_builder import sequence_to_peptide_graph_data

    return sequence_to_peptide_graph_data(sequence, sample_id=sample_id, properties=properties)


def iter_graph_records(path: Path) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    """读取一个现有 ``*.pkl.zst`` heavy atom graph 分片。"""

    from pipeline.graph.io import find_zstd, iter_pickle_zst

    zstd = find_zstd()
    yield from iter_pickle_zst(path, zstd)


def pick_graph_records(
    *,
    graph_root: Path | None = None,
    domains: list[str] | None = None,
    split: str = "train",
    max_files_per_domain: int = 1,
    max_records_per_domain: int = 2,
    max_nodes_per_record: int = 0,
) -> list[tuple[str, Path, int, dict[str, Any]]]:
    """从现有 graph root 中抽少量记录，供 smoke test 或小规模实验使用。"""

    paths = PipelinePaths()
    from pipeline.graph.io import pick_graph_records as pick_records

    return pick_records(
        graph_root or paths.graph_root,
        domains or ["small_molecule", "peptide", "protein", "dna", "rna"],
        split,
        max_files_per_domain,
        max_records_per_domain,
        max_nodes_per_record,
    )
