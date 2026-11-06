"""Shared metric reporting and semantic checkpoint lineage checks."""
import json
import math
import tempfile
from pathlib import Path
from moltoken.checkpoints import checkpoint_path_identity

def completed_lineage(run_root, task):
    """Read small training metadata, not weights; no batch/world-size/hash gates."""
    run_root = Path(run_root).resolve()
    result = {}
    for stage in ("alignment", "full"):
        best = (run_root / stage / "best").resolve()
        final = (run_root / stage / "final").resolve()
        state = json.loads((best / "trainer_state.json").read_text())
        final_state = json.loads((final / "trainer_state.json").read_text())
        config = json.loads((best / "axis_b_config.json").read_text())
        meta = config["metadata"]
        early = final_state["stats"]["early_stopping"]
        if (meta["task"] != task or meta["stage"] != stage
                or meta["checkpoint_role"] != "best" or state["stage"] != stage
                or final_state["stage"] != stage):
            raise ValueError(f"{stage}: task/stage/best checkpoint mismatch")
        if (meta["sampling_protocol"] != "single-task-repeated-stage-v1"
                or meta["natural_epoch_audit"]["maximum_epochs"] != 100):
            raise ValueError(f"{stage}: not the formal max100 repeated-stage protocol")
        completed, selected = early["completed_epochs"], early["best_epoch"]
        if (early["enabled"] is not True or not 1 <= selected <= completed <= 100
                or final_state["epoch"] != completed or state["epoch"] != selected
                or state["stats"]["early_stopping"]["best_epoch"] != selected):
            raise ValueError(f"{stage}: completed/selected early-stop epochs disagree")
        if early["stopped_early"]:
            if (completed < early["min_epochs"]
                    or early["epochs_without_improvement"] < early["patience"]):
                raise ValueError(f"{stage}: early-stop condition not met")
        elif completed != 100:
            raise ValueError(f"{stage}: neither early-stopped nor max100 complete")
        if checkpoint_path_identity(early["best_checkpoint"]) != checkpoint_path_identity(best):
            raise ValueError(f"{stage}: selected best checkpoint differs")
        loss = state["stats"]["eval"]["loss"]
        if not math.isfinite(loss) or loss != early["best_eval_loss"]:
            raise ValueError(f"{stage}: selected validation loss differs")
        if stage == "full" and checkpoint_path_identity(meta["resume_dir"]) != checkpoint_path_identity(run_root / "alignment/best"):
            raise ValueError("Full must resume this run's Alignment/best")
        result[stage] = {
            "checkpoint": str(best), "best_epoch": selected, "stopping_epoch": completed,
            "validation_loss": loss, "early_stopping": early,
            "data_root": meta["data_root"], "world_size": meta["world_size"],
            "batch_size": meta["natural_epoch_audit"]["batch_sizes_per_rank"][task],
        }
    if result["alignment"]["data_root"] != result["full"]["data_root"]:
        raise ValueError("Alignment/Full training data roots differ")
    return result


def number(value, *, percent=False):
    if value is None:
        return "—"
    if not math.isfinite(value):
        raise ValueError("nonfinite value in completed result")
    return f"{100*value:.4f}%" if percent else f"{value:.6f}"


def write_atomic(path, text):
    """Publish generated reports without exposing partially written Markdown."""
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=path.name+".", delete=False) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    temporary.chmod(path.stat().st_mode & 0o777 if path.exists() else 0o644)
    temporary.replace(path)

REFERENCE_LABEL = {"Escherichia coli": "Escherichia coli general", "Saccharomyces cerevisiae": "Saccharomyces cerevisiae", "Arabidopsis thaliana": "Arabidopsis thaliana", "Mus musculus": "Mus musculus", "Homo sapiens": "Homo sapiens"}
