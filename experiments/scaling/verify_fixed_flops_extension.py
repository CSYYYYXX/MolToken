#!/usr/bin/env python3
"""Verify completion artifacts and final domain metrics for a scaling matrix."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


DOMAIN_FIELDS = {
    "domain_samples",
    "domain_sample_fraction",
    "domain_input_tokens",
    "domain_input_token_fraction",
    "domain_loss_tokens",
    "domain_loss_token_fraction",
    "domain_loss_sum",
    "domain_loss_contribution",
    "domain_loss",
}
FRACTION_FIELDS = {
    "domain_sample_fraction",
    "domain_input_token_fraction",
    "domain_loss_token_fraction",
    "domain_loss_contribution",
}
EXPECTED_DOMAINS = {"small_molecule", "peptide", "protein", "dna", "rna"}


def last_train_event(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    last = None
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("event") == "train":
                last = event
    return last


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--matrix-json", type=Path, required=True)
    ap.add_argument("--expected-rows", type=int, default=0)
    ap.add_argument("--output-json", type=Path, default=None)
    args = ap.parse_args()

    matrix = json.loads(args.matrix_json.read_text(encoding="utf-8"))
    rows = list(matrix["rows"])
    issues: list[dict[str, Any]] = []
    verified: list[dict[str, Any]] = []
    seen_out_dirs: set[str] = set()

    if args.expected_rows and len(rows) != args.expected_rows:
        issues.append({"scope": "matrix", "issue": f"row count {len(rows)} != {args.expected_rows}"})

    for row in rows:
        key = f"{row['model_id']}/{row['flops_target_label']}"
        out_dir = Path(row["out_dir"])
        if str(out_dir) in seen_out_dirs:
            issues.append({"run": key, "issue": "duplicate out_dir", "out_dir": str(out_dir)})
            continue
        seen_out_dirs.add(str(out_dir))

        summary_path = out_dir / "summary.json"
        run_config_path = out_dir / "run_config.json"
        checkpoint_path = out_dir / "final" / "pytorch_model.bin"
        if not summary_path.exists():
            issues.append({"run": key, "issue": "missing summary", "path": str(summary_path)})
            continue
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception as exc:
            issues.append({"run": key, "issue": "invalid summary", "detail": str(exc)})
            continue

        global_step = int(summary.get("global_step", -1))
        max_steps = int(row["max_steps"])
        loss = summary.get("final_logged_train_loss")
        if global_step < max_steps:
            issues.append({"run": key, "issue": "incomplete step", "global_step": global_step, "max_steps": max_steps})
        if not checkpoint_path.exists():
            issues.append({"run": key, "issue": "missing final checkpoint", "path": str(checkpoint_path)})
        if not isinstance(loss, (int, float)) or not math.isfinite(float(loss)):
            issues.append({"run": key, "issue": "non-finite final training loss", "value": loss})

        if not run_config_path.exists():
            issues.append({"run": key, "issue": "missing run config", "path": str(run_config_path)})
            world_size = 0
        else:
            run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
            world_size = int(run_config.get("world_size", 0))
            if world_size <= 0:
                issues.append({"run": key, "issue": "invalid world size", "value": world_size})
        local_train_tokens = int(summary.get("total_train_loss_tokens", 0))
        global_train_tokens = local_train_tokens * world_size
        planned_tokens = int(row["actual_tokens"])
        token_relative_error = (global_train_tokens - planned_tokens) / planned_tokens if planned_tokens > 0 else math.inf
        if global_train_tokens <= 0:
            issues.append({"run": key, "issue": "non-positive global train tokens", "value": global_train_tokens})
        elif abs(token_relative_error) > 0.05:
            issues.append(
                {
                    "run": key,
                    "issue": "observed/planned token mismatch",
                    "global_train_tokens": global_train_tokens,
                    "planned_tokens": planned_tokens,
                    "relative_error": token_relative_error,
                }
            )

        event = last_train_event(out_dir / "train_log.jsonl")
        if event is None:
            issues.append({"run": key, "issue": "missing final train event"})
            continue
        missing_fields = sorted(DOMAIN_FIELDS - set(event))
        if missing_fields:
            issues.append({"run": key, "issue": "missing domain fields", "fields": missing_fields})
        invalid_domain_values = {}
        for field in DOMAIN_FIELDS & set(event):
            value = event[field]
            if not isinstance(value, dict):
                invalid_domain_values[field] = "not a mapping"
                continue
            observed_domains = set(value)
            if observed_domains != EXPECTED_DOMAINS:
                invalid_domain_values[field] = {
                    "missing_domains": sorted(EXPECTED_DOMAINS - observed_domains),
                    "unexpected_domains": sorted(observed_domains - EXPECTED_DOMAINS),
                }
                continue
            non_finite = {
                domain: item
                for domain, item in value.items()
                if not isinstance(item, (int, float)) or not math.isfinite(float(item))
            }
            if non_finite:
                invalid_domain_values[field] = {"non_finite": non_finite}
        if invalid_domain_values:
            issues.append({"run": key, "issue": "invalid domain values", "values": invalid_domain_values})
        bad_fraction_sums = {}
        for field in FRACTION_FIELDS & set(event):
            value = event[field]
            if not isinstance(value, dict):
                bad_fraction_sums[field] = "not a mapping"
                continue
            total = sum(float(item) for item in value.values())
            if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-6):
                bad_fraction_sums[field] = total
        if bad_fraction_sums:
            issues.append({"run": key, "issue": "invalid domain fraction sums", "values": bad_fraction_sums})

        verified.append(
            {
                "run": key,
                "global_step": global_step,
                "max_steps": max_steps,
                "final_training_loss": loss,
                "world_size": world_size,
                "global_train_loss_tokens": global_train_tokens,
                "planned_tokens": planned_tokens,
                "token_relative_error": token_relative_error,
                "checkpoint": str(checkpoint_path),
                "last_domain_step": event.get("step"),
            }
        )

    result = {
        "ok": not issues and len(verified) == len(rows),
        "matrix_json": str(args.matrix_json),
        "matrix_rows": len(rows),
        "verified_rows": len(verified),
        "issues": issues,
        "verified": verified,
    }
    output_path = args.output_json or args.matrix_json.parent / "completion_verification.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("ok", "matrix_rows", "verified_rows", "issues")}, ensure_ascii=False))
    raise SystemExit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
