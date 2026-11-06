"""Current formal downstream scope.

Dataset directory names, record task names, and raw benchmark names are
different identifiers. Keeping them separate prevents a retired task from
being re-enabled through a generic CLI argument by accident.
"""

from __future__ import annotations


ACTIVE_DATASET_IDS = (
    "unconditional_guacamol_v1",
    "unconditional_moses_v1",
)

ACTIVE_TRAIN_TASK_IDS = (
    "unconditional_guacamol",
    "unconditional_moses",
    "rfamgen_figure2_reconstruction17",
    "tcr_translate_pmhc_to_cdr3b",
    "dna_diffusion_celltype_generation",
    "pepbench_gen_length_conditioned",
    "dplm_length_conditioned_protein",
    "rabd_cdrh3_sequence_design",
    "codontransformer_protein_to_dna",
)

# The current MOSES/GuacaMol materialization intentionally keeps both datasets
# in one shared root.  Biological benchmarks use one independent root per
# task so their official/reconstructed splits and provenance do not get mixed.
# This mapping is the semantic data-scope check used by formal single-task SFT;
# it deliberately does not depend on whole-file hashes.
FORMAL_TASK_DATASET_SCOPES = {
    "unconditional_guacamol": ACTIVE_DATASET_IDS,
    "unconditional_moses": ACTIVE_DATASET_IDS,
    "rfamgen_figure2_reconstruction17": (
        "rfamgen_figure2_reconstruction17_v1",
    ),
    "tcr_translate_pmhc_to_cdr3b": (
        "tcr_translate_released_v1",
    ),
    "dna_diffusion_celltype_generation": (
        "dna_diffusion_official_fourlabel_v1",
    ),
    "pepbench_gen_length_conditioned": (
        "pepbench_gen_pretraining_exposed_v1",
    ),
    "dplm_length_conditioned_protein": (
        "dplm_uniref50_length100_400_partial_v1",
    ),
    "rabd_cdrh3_sequence_design": (
        "refinegnn_rabd_released_v1",
    ),
    "codontransformer_protein_to_dna": (
        "codontransformer_high_csi_trainpool_v1",
    ),
}

ACTIVE_UNCONDITIONAL_BENCHMARKS = (
    "guacamol",
    "moses",
)

# MOSES publishes train/test/test_scaffolds but no validation split.  Keep one
# deterministic percent of its training records for checkpoint selection;
# this never consults either official benchmark test split.
ACTIVE_VALIDATION_HOLDOUT = {
    "method": "sha256_record_id_fraction_v1",
    "tasks": {
        "unconditional_moses": {
            "fraction": 0.01,
            "seed": 20260827,
        }
    },
}


__all__ = [
    "ACTIVE_DATASET_IDS",
    "ACTIVE_TRAIN_TASK_IDS",
    "ACTIVE_UNCONDITIONAL_BENCHMARKS",
    "ACTIVE_VALIDATION_HOLDOUT",
    "FORMAL_TASK_DATASET_SCOPES",
]
