# Co-Adaptive Communication Evolution

Code accompanying the paper: **"Within-Lifetime Coupling, Not Inherited Learning: Minimal Conditions for Emergent Coordination in Independent Agent Populations"**

## Overview

This repository contains a minimal agent-based model demonstrating how independent populations of **producers** (signalers) and **receivers** can evolve a shared communication protocol through learning and selection, without pre-wired coordination.

## Requirements

```bash
pip install -r requirements.txt
```

- Python 3.8+
- NumPy
- Matplotlib
- Rich (for terminal UI)

## Quick Start

### Run a single experiment

```bash
python experiment.py
```

By default, this runs the full parameter sweep (360 experiments: 4 conditions × 3 noise levels × 30 seeds). Takes ~10 minutes on a modern multi-core machine.

### Generate figures

After running experiments:

```bash
python make_figures.py        # Generate Figure 1 (learning curves)
python make_figure2_simple.py # Generate Figure 2 (signal space)
python analyze_specialization.py # Generate Figure 3 (weight specialization)
```

## Key Parameters

Edit the top of `experiment.py` to modify:

- `GENERATIONS`: Number of evolutionary generations (default: 1000)
- `SIGNAL_DIM`: Signal dimensionality (default: 4, optimized via grid search)
- `LEARNING_RATE`: Within-lifetime learning rate (default: 0.02)
- `MUTATION_STD`: Evolutionary mutation rate (default: 0.02)
- `NOISE_SIGMA`: Channel noise level (default: 0.10)

## Experimental Conditions

- **main**: Both producers and receivers learn + evolve (co-adaptation)
- **no_comm**: Receivers ignore signals (baseline control)
- **fixed_mapping**: Producers frozen, only receivers adapt
- **shared**: Producers and receivers use identical weights (upper bound)

## Results

The main experiment demonstrates:
- **Co-evolution works**: Main condition achieves 0.873 ± 0.027 accuracy (8.7× above chance)
- **Controls fail**: Fixed/no-comm conditions remain at ~0.10 (chance level)
- **Noise degrades gracefully**: Performance drops with channel noise (0.87 → 0.56 → 0.53)

## File Structure

```
Experiment/
├── experiment.py                      # Main simulation code
├── make_figures.py                    # Generate Figure 1
├── make_figure2_simple.py             # Generate Figure 2
├── analyze_specialization.py          # Generate Figure 3 (run after sweep)
├── hyperparam_search.py               # Hyperparameter optimization (optional)
├── requirements.txt                   # Python dependencies
├── figure1_accuracy_curves.png        # Figure 1: Learning curves
├── figure2_signal_space.png           # Figure 2: Emergent signal clusters
├── figure3_specialization_heatmaps.png # Figure 3: Weight divergence
└── README.md
```

## Ablation experiments (`ablation/`)

Additional experiments supporting the paper, each self-contained (stdlib + numpy, fixed seeds):

- **`baldwinian_ablation.py`** -- Lamarckian vs Baldwinian inheritance (Table 2). Adds a genotype/phenotype split to test whether inheriting *learned* weights is necessary; the Baldwinian arm resets the phenotype to the genotype each generation.
- **`symmetric_rate_ablation.py`** -- symmetric vs asymmetric producer learning rule, crossed with both inheritance regimes (Table 3). Tests whether the producer's slower, success-only update is load-bearing for convergence (it is not).
- **`full_ablation.py`** -- unified harness combining the above plus a fitness-mode option (sum vs mean reward), used for the fitness-robustness check reported in the paper.

Each writes a `*_results.json` with per-condition accuracy (mean +/- 95% CI over seeds).

### Reproduce the genotype-only result with trial telemetry

Use Python 3.14 and the pinned replay dependency. Run from the repository root with a new output directory. The final argument is the number of worker processes. It can be omitted for serial execution.

```bash
python -m pip install -r ablation/requirements-replay.txt
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python ablation/replay_coadaptive.py replay-logs 8
```

The replay uses original ablation seeds 0–29, 1,000 generations, 500 training interactions per generation, and channel noise `σ = 0.1`. It saves 30 compressed logs with every training, learning-curve, and final decoding event. Each seed has 2,000 final evaluation attempts. Each attempt is paired with an independently seeded uniform guess at the same target context.

`coadaptive-final-telemetry.csv` contains 120,000 final decisions across both arms. Rows include the UTC observation timestamp, seed, generation, trial, arm, target, decoded context, and correctness. `replay-results.json` reports the final accuracy in each arm and the paired accuracy gain with a 95% t interval over the 30 seed-level gains. Wall-clock timestamps vary between runs.

### Fresh-seed experiment with mechanism controls

The [analysis plan](ablation/fresh-seed-run-plan.md) fixes seeds 1000–1029 and compares genotype-only inheritance with matched uniform guesses, learning without prior selection, selection with learning disabled, learned-weight inheritance, and a higher interaction budget. The standalone runner logs every final decision. The analyzer validates and assembles both trial CSVs and reports seed-level intervals and contrasts.

```bash
python -m pip install -r ablation/requirements-replay.txt
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python ablation/run_revision.py fresh-trials --workers 8
python ablation/analyze_revision.py fresh-trials fresh-analysis
```

Use Python 3.14 and new output directories. `fresh-analysis` contains the primary 120,000-row telemetry CSV, the 240,000-row comparison CSV, the per-seed table, and a JSON summary that includes the CSV checksums. Wall-clock timestamps vary between runs.

## Citation

If you use this code, please cite the accompanying paper:

*"Within-Lifetime Coupling, Not Inherited Learning: Minimal Conditions for Emergent Coordination in Independent Agent Populations"*

(Citation details will be added upon publication)

## License

MIT License - see LICENSE file for details

## Reproducibility

All experiments use fixed random seeds. Running the full parameter sweep with default settings reproduces the exact results reported in the paper.

- Random seed: 42 (for single runs)
- Sweep seeds: 100-129 (30 independent runs per condition)

## Questions or Issues

Please open a GitHub issue for questions, bug reports, or suggestions.

