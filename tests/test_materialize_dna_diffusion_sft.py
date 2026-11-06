from __future__ import annotations

import unittest

from cli.data.materialize_dna_diffusion_sft import (
    LABELS,
    exact_stratified_inner_validation,
)


class DnaDiffusionSplitTest(unittest.TestCase):
    def test_inner_validation_is_exact_stratified_and_deterministic(self) -> None:
        rows = [
            {"id": f"{label}-{index}", "cell_type": label}
            for label in LABELS
            for index in range(100)
        ]
        first = exact_stratified_inner_validation(rows, fraction=0.05, seed=42)
        second = exact_stratified_inner_validation(rows, fraction=0.05, seed=42)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 20)
        for label in LABELS:
            self.assertEqual(sum(value.startswith(label + "-") for value in first), 5)

    def test_invalid_fraction_fails(self) -> None:
        rows = [{"id": f"{label}-0", "cell_type": label} for label in LABELS]
        with self.assertRaises(ValueError):
            exact_stratified_inner_validation(rows, fraction=0.0, seed=42)


if __name__ == "__main__":
    unittest.main()
