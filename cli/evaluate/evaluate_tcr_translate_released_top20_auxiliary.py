#!/usr/bin/env python3
"""Score the released top-20 TCR-TRANSLATE auxiliary validation track.

The numerical sequence metrics are delegated to the released
``src.evaluation.ModelEvaluator`` static helpers.  This adapter does not
invent an offline perplexity: the official implementation obtains perplexity
from its own model/tokenizer cross-entropy, which is not comparable to Axis
FDDFS-BPE likelihood and cannot be reconstructed from prediction JSONL.
"""

from __future__ import annotations

from pipeline import paths as _moltoken_paths

import argparse
import csv
import json
from pathlib import Path
import socket
import sys
from typing import Any, Mapping, Sequence


PIPELINE_ROOT = Path(__file__).resolve().parents[2]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from pipeline.runtime_host import require_compute_host  # noqa: E402


TRACK = "released_top20_validation_auxiliary"
PREDICTION_PROTOCOL = "axis-tcr-translate-released-top20-auxiliary-predictions-v1-20260830"
EXPECTED_GROUPS = 20
OFFICIAL_K = 100
OFFICIAL_NUM_BEAMS = 300
OFFICIAL_MAX_NEW_TOKENS = 25
OFFICIAL_NO_REPEAT_NGRAM_SIZE = 5
DEFAULT_OFFICIAL_REPO = Path(
    str(_moltoken_paths.LOCAL_DATA_ROOT / 'tools/tcr_translate_official/dhuvik-tcr_translate-6853e5c')
)


def require_worker() -> None:
    require_compute_host("TCR-TRANSLATE metric evaluation", socket.gethostname())


def _string_list(value: Any, *, label: str, allow_empty: bool) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    values: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or (not allow_empty and not item):
            raise ValueError(f"{label}[{index}] is not a valid string")
        values.append(item)
    return values


def validate_prediction(record: Any, *, line_number: int) -> dict[str, Any]:
    label = f"prediction line {line_number}"
    if not isinstance(record, dict):
        raise ValueError(f"{label}: record must be an object")
    if record.get("protocol") != PREDICTION_PROTOCOL:
        raise ValueError(f"{label}: unexpected protocol")
    if record.get("track") != TRACK or record.get("headline") is not False:
        raise ValueError(f"{label}: result is not the released top-20 auxiliary track")
    generation = record.get("generation")
    if not isinstance(generation, dict):
        raise ValueError(f"{label}: generation must be an object")
    expected = {
        "mode": "beam",
        "k": OFFICIAL_K,
        "num_beams": OFFICIAL_NUM_BEAMS,
        "max_new_tokens": OFFICIAL_MAX_NEW_TOKENS,
        "no_repeat_ngram_size": OFFICIAL_NO_REPEAT_NGRAM_SIZE,
        "do_sample": False,
        "invalid_or_duplicate_filtering": False,
        "replacement_sampling": False,
    }
    for key, expected_value in expected.items():
        if generation.get(key) != expected_value:
            raise ValueError(
                f"{label}: generation.{key}={generation.get(key)!r}, "
                f"expected {expected_value!r}"
            )
    translations = _string_list(
        record.get("translations"), label=f"{label}.translations", allow_empty=True
    )
    if len(translations) != OFFICIAL_K:
        raise ValueError(
            f"{label}: expected {OFFICIAL_K} translations, got {len(translations)}"
        )
    references = _string_list(
        record.get("reference_translations"),
        label=f"{label}.reference_translations",
        allow_empty=False,
    )
    if not references:
        raise ValueError(f"{label}: reference list is empty")
    greedy = record.get("greedy_translation")
    if not isinstance(greedy, str):
        raise ValueError(f"{label}: greedy_translation must be a string")
    candidates = record.get("beam_candidates")
    if not isinstance(candidates, list) or len(candidates) != OFFICIAL_K:
        raise ValueError(f"{label}: beam_candidates must have {OFFICIAL_K} slots")
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            raise ValueError(f"{label}: candidate {index} is not an object")
        if candidate.get("rank") != index + 1:
            raise ValueError(f"{label}: candidate rank order changed at {index}")
        if candidate.get("official_translation") != translations[index]:
            raise ValueError(
                f"{label}: candidate and translation disagree at rank {index + 1}"
            )
    pmhc = record.get("pmhc")
    if not isinstance(pmhc, dict) or not isinstance(pmhc.get("official_key"), str):
        raise ValueError(f"{label}: pmhc.official_key is required")
    return record


def load_predictions(
    path: Path, *, expected_groups: int = EXPECTED_GROUPS
) -> list[dict[str, Any]]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"prediction JSONL missing: {path}")
    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_pmhc: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                raise ValueError(f"{path}:{line_number}: blank line")
            record = validate_prediction(json.loads(raw), line_number=line_number)
            record_id = record.get("id")
            if not isinstance(record_id, str) or not record_id:
                raise ValueError(f"{path}:{line_number}: missing id")
            pmhc_key = record["pmhc"]["official_key"]
            if record_id in seen_ids or pmhc_key in seen_pmhc:
                raise ValueError(
                    f"{path}:{line_number}: duplicate id or pMHC key"
                )
            seen_ids.add(record_id)
            seen_pmhc.add(pmhc_key)
            records.append(record)
    if len(records) != expected_groups:
        raise ValueError(
            f"prediction group count changed: expected={expected_groups} "
            f"actual={len(records)}"
        )
    return records


def load_released_evaluator(official_repo: Path) -> Any:
    official_repo = official_repo.expanduser().resolve()
    evaluator_path = official_repo / "src/evaluation.py"
    if not evaluator_path.is_file():
        raise FileNotFoundError(f"released evaluator missing: {evaluator_path}")
    loaded_src = sys.modules.get("src")
    if loaded_src is not None:
        location = Path(str(getattr(loaded_src, "__file__", ""))).resolve()
        try:
            location.relative_to(official_repo)
        except ValueError as exc:
            raise RuntimeError(
                "top-level 'src' is already imported from another repository; "
                "run the official metric adapter in a fresh process"
            ) from exc
    sys.path.insert(0, str(official_repo))
    # The released repository imports the legacy top-level ``LogitsWarper``
    # symbol even though this metric-only adapter calls only ModelEvaluator's
    # static sequence helpers.  Transformers 4.57 retains the same callable
    # base as LogitsProcessor but no longer exports LogitsWarper.  Provide the
    # compatibility name in-process rather than editing the released source.
    import transformers

    if not hasattr(transformers, "LogitsWarper"):
        from transformers.generation.logits_process import LogitsProcessor

        transformers.LogitsWarper = LogitsProcessor
    from src.evaluation import ModelEvaluator

    return ModelEvaluator


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty metric list")
    return float(sum(float(value) for value in values) / len(values))


def compute_official_sequence_metrics(
    records: Sequence[Mapping[str, Any]], evaluator: Any
) -> dict[str, Any]:
    atomic: list[dict[str, Any]] = []
    all_translations: list[str] = []
    greedy_translations: list[str] = []
    reference_lists: list[list[str]] = []
    for record in records:
        translations = list(record["translations"])
        references = list(record["reference_translations"])
        greedy = str(record["greedy_translation"])
        precision = float(evaluator._precision_at_k(translations, references, k=OFFICIAL_K))
        recall = float(evaluator._recall_at_k(translations, references, k=OFFICIAL_K))
        f1 = float(evaluator._f1_at_k(translations, references, k=OFFICIAL_K))
        item = {
            "pMHC": record["pmhc"]["official_key"],
            "char-bleu": float(
                evaluator._sequence_bleu(
                    greedy,
                    references,
                    max_references=20,
                    max_ngram=4,
                )
            ),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "d_edit": float(evaluator._mean_edit_distance(translations, references)),
            "seq_recovery": float(
                evaluator._mean_sequence_recovery(translations, references)
            ),
            "translations": translations,
            "reference_translations": references,
        }
        atomic.append(item)
        all_translations.extend(translations)
        greedy_translations.append(greedy)
        reference_lists.append(references)
    dataset = {
        "char-bleu": float(
            evaluator._dataset_bleu(
                greedy_translations,
                reference_lists,
                max_references=20,
                max_ngram=4,
            )
        ),
        "precision": _mean([item["precision"] for item in atomic]),
        "recall": _mean([item["recall"] for item in atomic]),
        "f1": _mean([item["f1"] for item in atomic]),
        "d_edit": _mean([item["d_edit"] for item in atomic]),
        "seq_recovery": _mean([item["seq_recovery"] for item in atomic]),
        "diversity": len(set(all_translations)) / len(all_translations),
        "perplexity": None,
    }
    return {
        "protocol": PREDICTION_PROTOCOL,
        "track": TRACK,
        "headline": False,
        "official_evaluator_scope": "released sequence metrics called directly",
        "dataset_metrics": dataset,
        "atomic_metrics": atomic,
        "denominators": {
            "pmhc_groups": len(records),
            "translations_per_pmhc": OFFICIAL_K,
            "translation_slots": len(all_translations),
            "reference_rows_including_duplicates": sum(
                len(values) for values in reference_lists
            ),
        },
        "perplexity_status": {
            "available": False,
            "reason": (
                "released perplexity is cross-entropy under the evaluated model's "
                "own tokenizer; it is unavailable from decoded predictions and not "
                "cross-tokenizer comparable to Axis FDDFS-BPE"
            ),
        },
    }


def write_outputs(
    result: Mapping[str, Any], *, output_prefix: Path, model_name: str
) -> dict[str, str]:
    output_prefix = output_prefix.expanduser().resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": Path(str(output_prefix) + ".json"),
        "dataset_csv": Path(str(output_prefix) + ".dataset.csv"),
        "atomic_csv": Path(str(output_prefix) + ".atomic.csv"),
    }
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite metric outputs: {existing}")
    paths["json"].write_text(
        json.dumps(
            {**result, "model": model_name},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    dataset_fields = [
        "model",
        "char-bleu",
        "precision",
        "recall",
        "f1",
        "d_edit",
        "seq_recovery",
        "diversity",
        "perplexity",
    ]
    with paths["dataset_csv"].open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=dataset_fields)
        writer.writeheader()
        writer.writerow({"model": model_name, **result["dataset_metrics"]})
    atomic_fields = [
        "pMHC",
        "char-bleu",
        "precision",
        "recall",
        "f1",
        "d_edit",
        "seq_recovery",
        "translations",
        "reference_translations",
        "model",
    ]
    with paths["atomic_csv"].open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=atomic_fields)
        writer.writeheader()
        for item in result["atomic_metrics"]:
            writer.writerow(
                {
                    **item,
                    "translations": repr(item["translations"]),
                    "reference_translations": repr(item["reference_translations"]),
                    "model": model_name,
                }
            )
    return {key: str(path) for key, path in paths.items()}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--official-repo", type=Path, default=DEFAULT_OFFICIAL_REPO)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    require_worker()
    records = load_predictions(args.predictions)
    evaluator = load_released_evaluator(args.official_repo)
    result = compute_official_sequence_metrics(records, evaluator)
    outputs = write_outputs(
        result,
        output_prefix=args.output_prefix,
        model_name=args.model_name,
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "outputs": outputs,
                "dataset_metrics": result["dataset_metrics"],
                "perplexity_status": result["perplexity_status"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
