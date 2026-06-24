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

