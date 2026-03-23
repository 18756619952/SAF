# Compliance Dynamics and Policy-Economic Implications of Sustainable Aviation Fuel under Deep Energy Transition

This repository contains the code used to analyze how electricity and hydrogen transitions reshape the life-cycle carbon intensity, production cost, compliance windows, and policy-economic outcomes of sustainable aviation fuel (SAF) pathways under an LCFS-like policy framework.

The workflow combines pathway-level LCA and TEA calculations with NGFS-based background drivers, policy-layer analysis, and figure generation for the manuscript and Supplementary Information.

## Repository structure

- `model/` – core pathway-level LCA and TEA models
- `scenario/` – scripts for building NGFS-based background drivers
- `analysis/` – policy-layer analysis, including compliance windows and related metrics
- `visualization/` – scripts for generating main-text and supplementary figures
- `workflow/` – workflow runners for coupled execution and batch processing

## Main scripts

- `model/lca_model.py` – computes pathway-level life-cycle GHG emissions on a 1 MJ SAF basis
- `model/tea_model.py` – computes pathway-level production costs under the manuscript TEA boundary
- `workflow/run_coupled_lca_tea.py` – runs the LCA and TEA models in sequence
- `scenario/build_ngfs_drivers.py` – builds scenario-dependent background electricity and hydrogen drivers
- `workflow/run_dynamic_ngfs_batch.py` – applies scenario-specific drivers across regions, scenarios, and years
- `analysis/policy_layer_analyze.py` – translates CI and cost trajectories into policy-layer metrics
- `visualization/make_paper_figures.py` – generates main manuscript figures
- `visualization/make_extra_figures.py` – generates additional / supplementary figures
- `workflow/run_all.py` – top-level pipeline runner

## Workflow overview

The codebase follows the workflow below:

1. Build scenario-dependent background drivers (`scenario/build_ngfs_drivers.py`)
2. Run pathway-level LCA and TEA models across NGFS cases (`workflow/run_dynamic_ngfs_batch.py`)
3. Perform policy-layer analysis (`analysis/policy_layer_analyze.py`)
4. Generate manuscript and supplementary figures (`visualization/make_paper_figures.py`, `visualization/make_extra_figures.py`)

For an end-to-end run, use:

```bash
python workflow/run_all.py
