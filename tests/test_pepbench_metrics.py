import numpy as np
import pytest

from pipeline.benchmarks.pepbench_metrics import (
    diversity_metric, embedding_precision_recall, frechet_distance, marginal_jsd,
    novelty_metrics, sequence_is_valid, slot_metrics,
)


def test_validity_and_duplicate_denominators():
    rows = [dict(id=str(i), status="valid", decoded_sequence="AAAAA", condition="5") for i in range(2)]
    rows += [dict(id="2", status="missing_eos", decoded_sequence=None, condition="5")]
    metrics, seq = slot_metrics(rows)
    assert metrics["validity"] == 2 / 3
    assert metrics["uniqueness_within_valid"] == .5
    assert metrics["exact_length_rate"] == 2 / 3
    assert len(seq) == 2
    assert not sequence_is_valid("AAAA")
    assert not sequence_is_valid("AAAAAX")
    assert not sequence_is_valid("aAAAA")
    with pytest.raises(ValueError, match="duplicate"):
        slot_metrics(rows + [rows[0]])


def test_copy_train_zero_novelty_duplicate_diversity_zero():
    seq = ["AAAAA"] * 3
    result = novelty_metrics(seq, ["AAAAA", "CCCCCCC"])
    assert result["exact_novelty_within_valid"] == 0
    assert result["nearest_normalized_edit_distance"]["mean"] == 0
    assert diversity_metric(seq)["mean_normalized_edit_distance"] == 0
    assert diversity_metric(["AAAAA"])["mean_normalized_edit_distance"] is None


def test_nearest_distance_normalizes_each_pair_and_keeps_duplicates():
    result = novelty_metrics(["AAAAA", "AAAAC", "AAAAC"], ["AAAAA", "AAAAAA"], reference_batch=1)
    assert result["exact_novelty_within_valid"] == 2 / 3
    # AAAAC -> AAAAA is 1/5, AAAAC -> AAAAAA is 2/6.
    assert result["nearest_normalized_edit_distance"]["mean"] == pytest.approx((0 + .2 + .2) / 3)


def test_empty_valid_outputs_are_explicitly_undefined():
    result = novelty_metrics([], ["AAAAA"])
    assert result["exact_novelty_within_valid"] is None
    assert marginal_jsd([], ["AAAAA"])["aa_composition_jsd_bits"] is None
    assert frechet_distance(np.ones((4, 2)), np.empty((0, 2))) is None


def test_distribution_identity_and_separation():
    points = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.], [.5, .5]])
    assert frechet_distance(points, points) == pytest.approx(0, abs=1e-8)
    assert frechet_distance(points, points + 10) == pytest.approx(200)
    assert embedding_precision_recall(points, points, batch_size=2) == {"precision": 1., "recall": 1., "k": 3}
    assert embedding_precision_recall(points, points + 10, batch_size=2) == {"precision": 0., "recall": 0., "k": 3}
    assert embedding_precision_recall(points, points[:3])["precision"] is None
    assert marginal_jsd(["AAAAACCCCC"], ["CCCCCAAAAA"])["aa_composition_jsd_bits"] == pytest.approx(0)
    assert marginal_jsd(["AAAAA"], ["CCCCC"])["aa_composition_jsd_bits"] == pytest.approx(1)


def test_diversity_subset_is_repeatable_and_does_not_deduplicate():
    sequences = ["AAAAA", "CCCCC", "DDDDD", "EEEEE"] * 5
    one = diversity_metric(sequences, seed=42, maximum_slots=7)
    assert one == diversity_metric(sequences, seed=42, maximum_slots=7)
    assert len(one["valid_slot_indices"]) == 7
    assert one["pairs"] == 21


def test_bad_embeddings_fail_loudly():
    values = np.ones((4, 2))
    values[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        frechet_distance(values, np.ones((4, 2)))


def test_eisenberg_moment_matches_published_modlamp_example():
    from cli.evaluate.evaluate_pepbench_gen import physicochemical
    # https://modlamp.org/modlamp.html PeptideDescriptor.calculate_moment
    result = physicochemical(["GLFDIVKKVVGALGSL"])
    assert result["hydrophobic_moment_alpha"][0] == pytest.approx(.48790226, abs=1e-8)
