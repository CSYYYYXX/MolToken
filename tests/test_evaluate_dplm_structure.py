import sys
from types import SimpleNamespace

import pytest
from cli.evaluate.evaluate_dplm_structure import CONDITIONS, TASK, native_folder, score_rows, validate_rows


def fixture_rows():
    rows = []
    for condition in CONDITIONS:
        rows.extend([
            {"id": f"{condition}:0", "condition": condition, "sample_index": 0, "task": TASK,
             "status": "valid", "ended_with_eos": True, "decoded_sequence": "A" * int(condition)},
            {"id": f"{condition}:1", "condition": condition, "sample_index": 1, "task": TASK,
             "status": "valid", "ended_with_eos": True, "decoded_sequence": "A" * 100},
            {"id": f"{condition}:2", "condition": condition, "sample_index": 2, "task": TASK,
             "status": "missing_eos", "ended_with_eos": False, "decoded_sequence": None}])
    return rows


def test_raw_denominators_duplicates_and_wrong_length(tmp_path):
    rows = fixture_rows()
    validate_rows(rows, per_condition=3)
    calls = []
    def fold(sequence):
        calls.append(sequence)
        return {"plddt": len(sequence) / 10, "ptm": .5, "pdb": "ATOM synthetic fixture\n"}
    result = score_rows(rows, tmp_path / "score", fold)
    assert len(calls) == 4
    assert result["requested"] == 12 and result["folded"] == 8
    assert result["fold_coverage"] == pytest.approx(2 / 3)
    assert result["per_condition"]["200"]["plddt_valid"]["mean"] == 15
    assert result["per_condition"]["200"]["plddt_exact_length_only"]["mean"] == 20
    assert result["per_condition"]["200"]["exact_length_coverage"] == pytest.approx(1 / 3)
    assert result["structure_suite_complete"] is False


def test_native_error_is_not_relabelled_as_invalid_generation(tmp_path):
    def fail(_sequence):
        raise RuntimeError("native evaluator error")
    with pytest.raises(RuntimeError, match="native evaluator error"):
        score_rows(fixture_rows(), tmp_path / "score", fail)
    assert not (tmp_path / "score/summary.json").exists()


def test_slot_grid_and_valid_sequence_are_checked():
    rows = fixture_rows()
    with pytest.raises(ValueError, match="expected 40"):
        validate_rows(rows, per_condition=40)
    rows[1]["sample_index"] = 0
    with pytest.raises(ValueError, match="indices"):
        validate_rows(rows)
    rows = fixture_rows()
    rows[0]["decoded_sequence"] = "ABC"
    with pytest.raises(ValueError, match="canonical"):
        validate_rows(rows)


@pytest.mark.parametrize("device,force_float32,expected_dtype", [
    ("cpu", False, "torch.float32"),
    ("cuda", False, "torch.float16"),
    ("cuda", True, "torch.float32"),
])
def test_native_precision_option_preserves_existing_paths(tmp_path, monkeypatch, device, force_float32, expected_dtype):
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    for name in ("esmfold_3B_v1", "esm2_t36_3B_UR50D", "esm2_t36_3B_UR50D-contact-regression"):
        (checkpoints / (name + ".pt")).write_bytes(b"test asset")
    calls = []
    parameter = SimpleNamespace(dtype="torch.float16")

    def to_float():
        parameter.dtype = "torch.float32"

    model = SimpleNamespace(
        esm=SimpleNamespace(float=to_float, parameters=lambda: iter([parameter])),
        cfg=SimpleNamespace(trunk=SimpleNamespace(max_recycles=4)),
        cpu=lambda: calls.append("cpu"), cuda=lambda: calls.append("cuda"),
        set_chunk_size=lambda value: calls.append(("chunk", value)),
    )
    model.eval = lambda: model
    fake_torch = SimpleNamespace(set_num_threads=lambda value: calls.append(("threads", value)),
                                 hub=SimpleNamespace(set_dir=lambda path: None))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "esm", SimpleNamespace(
        __file__="native/esm.py", pretrained=SimpleNamespace(esmfold_v1=lambda: model)))
    _, metadata = native_folder(tmp_path, device=device, esm_float32=force_float32, cpu_threads=1)
    assert metadata["esm_dtype"] == expected_dtype
    assert metadata["num_recycles_argument"] is None
    assert metadata["cpu_threads"] == 1
    assert calls == [("threads", 1), device, ("chunk", 64)]
