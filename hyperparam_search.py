#!/usr/bin/env python3
"""
Quick hyperparameter search for optimal settings.
Runs shorter experiments (200 gens) to find promising configurations.
"""

import sys
import json
from experiment import run_sim, console

# Test grid
LEARNING_RATES = [0.005, 0.01, 0.02, 0.03]
MUTATION_STDS = [0.005, 0.01, 0.02]
SIGNAL_DIMS = [2, 3, 4]  # try higher dimensional signals

# Run shorter experiments for speed
TEST_GENERATIONS = 200
NOISE_SIGMA = 0.10
CONDITION = "main"

results = []

console.print("\n[bold cyan]HYPERPARAMETER SEARCH[/]")
console.print(f"Testing {len(LEARNING_RATES)} × {len(MUTATION_STDS)} × {len(SIGNAL_DIMS)} = {len(LEARNING_RATES) * len(MUTATION_STDS) * len(SIGNAL_DIMS)} configs")
console.print(f"Each run: {TEST_GENERATIONS} generations\n")

config_num = 0
total_configs = len(LEARNING_RATES) * len(MUTATION_STDS) * len(SIGNAL_DIMS)

for lr in LEARNING_RATES:
    for mut in MUTATION_STDS:
        for sig_dim in SIGNAL_DIMS:
            config_num += 1
            
            # Override globals (bit hacky but works)
            import experiment
            experiment.LEARNING_RATE = lr
            experiment.MUTATION_STD = mut
            experiment.SIGNAL_DIM = sig_dim
            experiment.GENERATIONS = TEST_GENERATIONS
            
            console.print(f"[yellow][{config_num}/{total_configs}] Testing: η={lr:.3f}, σ_m={mut:.3f}, dim={sig_dim}[/]", end=" ")
            
            result = run_sim(
                condition=CONDITION,
                noise_sigma=NOISE_SIGMA,
                seed=42,
                show_plots=False,
                verbose=False
            )
            
            results.append({
                'learning_rate': lr,
                'mutation_std': mut,
                'signal_dim': sig_dim,
                'final_acc': result['final_acc'],
                'final_mi': result['final_mi'],
                'final_effectiveness': result['final_effectiveness'],
            })
            
            console.print(f"→ acc={result['final_acc']:.3f}, MI={result['final_mi']:.2f}, E={result['final_effectiveness']:.2f}")

# Sort by final accuracy
results_sorted = sorted(results, key=lambda x: x['final_acc'], reverse=True)

console.print("\n[bold green]TOP 5 CONFIGURATIONS (by accuracy):[/]\n")
for i, r in enumerate(results_sorted[:5]):
    console.print(
        f"{i+1}. η={r['learning_rate']:.3f}, σ_m={r['mutation_std']:.3f}, dim={r['signal_dim']} → "
        f"acc={r['final_acc']:.3f}, MI={r['final_mi']:.2f}, E={r['final_effectiveness']:.2f}"
    )

# Save results
with open("hyperparam_search_results.json", "w") as f:
    json.dump(results_sorted, f, indent=2)

console.print("\n[cyan]✓ Results saved to hyperparam_search_results.json[/]")
console.print(f"\nBest config: η={results_sorted[0]['learning_rate']:.3f}, "
              f"σ_m={results_sorted[0]['mutation_std']:.3f}, "
              f"dim={results_sorted[0]['signal_dim']}")

