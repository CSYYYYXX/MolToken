# Benchmark evaluation

The evaluation modules handle data conversion, generation and scoring. Reference
datasets, ChemNet/ESM/folding weights and external evaluation repositories are
downloaded separately. All paths below are relative to the repository root.

## MOSES and GuacaMol distribution learning

Use dedicated evaluator environments. Install MolToken and its training
dependencies, then the appropriate `requirements/moses.txt` or
`requirements/guacamol.txt`. Pin compatible PyTorch/CUDA separately. GuacaMol's
FCD 1.1 is the Keras/H5 implementation; FCD 1.2 or PyTorch FCD is not a drop-in
replacement for that protocol. MOSES uses `molsets==1.0` and PyTorch ChemNet.

MOSES reference source: `molecularsets/moses`, commit
`dd7ed6ab38e23afd3ef5371d67939a1760bd8599`. The published CSV URL is recorded in
`cli/data/stage_moses_references.py` as `OFFICIAL_COMBINED_URL`. Stage its
train/test/test-scaffold rows without re-splitting the official tests:

```bash
python -m cli.data.stage_moses_references \
  --combined ./data/moses/dataset_v1.csv --output-root ./data/moses/references

moltoken moses --checkpoint ./runs/moses/full/best --output-root ./results/moses \
  --text-encoder ./weights/all-mpnet-base-v2 --chemnet ./weights/ChemNet_v0.13_pretrained.pt \
  --train ./data/moses/references/train.csv --test ./data/moses/references/test.csv \
  --test-scaffolds ./data/moses/references/test_scaffolds.csv \
  --test-stats ./data/moses/test_stats.npz \
  --test-scaffolds-stats ./data/moses/test_scaffolds_stats.npz
```

The MOSES adapter uses 30,000 raw draws per seed (42, 43, 44), keeps invalid
draws in the denominator and reports validity, uniqueness, novelty, filters,
FCD, SNN, fragment/scaffold similarity, internal diversity and property
distances through the official evaluator. Metric details and sample accounting
are recorded in its output JSON. Invalid outputs are not resampled.

GuacaMol reference source: `BenevolentAI/guacamol`, commit
`60ebe1f6a396f16e08b834dce448e9343d259feb`. The adapter verifies the published
0.5.5 distribution-learning implementation and its ChemNet asset. Supply
the downloaded source, wheels and H5 file explicitly:

```bash
moltoken guacamol --checkpoint ./runs/guacamol/full/best --output ./results/guacamol.json \
  --training-file ./data/guacamol/guacamol_v1_train.smiles \
  --text-encoder ./weights/all-mpnet-base-v2 \
  --official-source-root ./external/guacamol \
  --official-guacamol-wheel ./wheels/guacamol-0.5.5-py3-none-any.whl \
  --official-fcd-wheel ./wheels/FCD-1.1-py3-none-any.whl \
  --official-chemnet-h5 ./weights/ChemNet_v0.13_pretrained.h5
```

Validity, uniqueness, novelty, KL-divergence score and FCD score are delegated
to GuacaMol's distribution-learning suite. Do not compare its transformed FCD
score directly with MOSES's raw FCD distance. Check `--help` for sampling,
explicit device selection, seeds, provenance and output options.

## Biological / cross-domain adapters

The following task-specific modules are also included. Their protocol IDs and
reference files specify the evaluation setup; some reproduce a paper's workflow
instead of calling a packaged evaluator.

| Protocol | Data conversion (`cli.data`) | Sampling / evaluation (`cli.evaluate`) | External components |
|---|---|---|---|
| DNA-Diffusion cell-type generation | `materialize_dna_diffusion_sft` | `sample_axis_b_biological_conditions`, `evaluate_dna_motifs`, `evaluate_dna_diffusion_representation` | released DNA-Diffusion data, motif references/tools and representation assets |
| DPLM length-conditioned protein generation | `materialize_dplm_length_conditioned_protein_sft`, `materialize_dplm_natural_reference` | `evaluate_dplm_structure`, `evaluate_dplm_inner_tm`, `evaluate_dplm_natural_reference` | pretrained folding model, alignment/structure-scoring executables, natural references |
| RfamGen Figure 2 reconstruction | `materialize_rfamgen_figure2_g6`, `materialize_rfamgen_figure2_axis`, `materialize_rfamgen_figure2_length_pools` | `evaluate_rfamgen_figure2_reconstruction17` | Rfam sequences/covariance models, Infernal, ViennaRNA and NLTK |
| PepBench generation | `materialize_pepbench_gen_sft`, `prepare_pepbench_generation_conditions`, `build_aa_source_index` | `evaluate_pepbench_gen`, source-distance/novelty scorers, `summarize_pepbench_gen` | selected source pool, ESM checkpoint, RapidFuzz |
| TCR-TRANSLATE | `materialize_tcr_translate_sft`, external-13/top-20 reference materializers | matching `sample_tcr_translate_*` and `evaluate_tcr_translate_*` | released sequence data and official TCR-TRANSLATE scoring source |
| RAbD CDR-H3 design | `materialize_rabd_cdrh3_sft` | `evaluate_rabd_sequence_design`, `evaluate_refinegnn_rabd_native`, `summarize_rabd_sequence_design` | RefineGNN released test cases and repository for native baseline |
| CodonTransformer | `materialize_codon_sft`, `prepare_codon_inference`, `prepare_codon_genome_reference` | `sample_codon_sequence_design`, `score_codon_sequence_design`, R helpers | released data, organism codon references, Biopython and R metric dependencies |

`pip install '.[bio-eval]'` installs common Python helpers; it does not download
model weights, official repositories, R packages or scientific executables.
Each adapter's `--help` lists the required local paths. The official TCR evaluator
runs in a separate process to avoid a conflict between its `src` package and
MolToken's `src` namespace.

General inverse decoding is available through `moltoken decode` and
`moltoken.MolTokenizer`.
