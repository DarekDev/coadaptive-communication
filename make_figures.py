#!/usr/bin/env python3
"""
Generate Figure 1 and fill in paper values from sweep results.
Run this AFTER the full parameter sweep completes.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

print("Loading results...")
with open("sweep_results_full.pkl", "rb") as f:
    all_results = pickle.load(f)

print(f"Loaded {len(all_results)} runs")

# Group by (condition, noise_sigma)
grouped = defaultdict(list)
for r in all_results:
    key = (r['condition'], r['noise_sigma'])
    grouped[key].append(r)

# Plot settings
plt.style.use("dark_background")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Figure 1: Alignment Accuracy Across Generations", fontsize=16, fontweight='bold')

# Define plot order
plot_configs = [
    ("main", 0.10, "Main: Low Noise (σ=0.1)", axes[0, 0]),
    ("main", 0.50, "Main: High Noise (σ=0.5)", axes[0, 1]),
    ("shared", 0.10, "Shared Weights (upper bound)", axes[1, 0]),
    ("no_comm", 0.10, "Controls (σ=0.1)", axes[1, 1]),
]

# Plot each condition
for condition, sigma, title, ax in plot_configs:
    key = (condition, sigma)
    
    if key in grouped:
        runs = grouped[key]
        
        # Extract histories
        histories = [r['acc_history'] for r in runs]
        histories = np.array(histories)  # shape: (n_seeds, n_generations)
        
        # Compute mean and 95% CI
        mean_curve = histories.mean(axis=0)
        std_curve = histories.std(axis=0, ddof=1)
        n = len(runs)
        ci_95 = 1.96 * std_curve / np.sqrt(n)
        
        generations = np.arange(1, len(mean_curve) + 1)
        
        # Plot
        ax.plot(generations, mean_curve, lw=2, label=f"{condition}, σ={sigma}", color='cyan')
        ax.fill_between(generations, mean_curve - ci_95, mean_curve + ci_95, 
                        alpha=0.3, color='cyan')
        
        # If it's the controls panel, add other baselines
        if title.startswith("Controls"):
            # Add fixed_mapping
            key_fixed = ("fixed_mapping", sigma)
            if key_fixed in grouped:
                runs_fixed = grouped[key_fixed]
                histories_fixed = np.array([r['acc_history'] for r in runs_fixed])
                mean_fixed = histories_fixed.mean(axis=0)
                ax.plot(generations, mean_fixed, lw=2, label="Fixed mapping", 
                       color='orange', linestyle='--')
            
            # no_comm is already plotted as main line
            ax.plot(generations, mean_curve, lw=2, label="No communication", color='red')
        
        ax.set_ylim(0, 1)
        ax.set_xlabel("Generation")
        ax.set_ylabel("Accuracy")
        ax.set_title(title)
        ax.legend(loc='lower right')
        ax.grid(alpha=0.2)

plt.tight_layout()
plt.savefig("figure1_accuracy_curves.png", dpi=300, bbox_inches='tight')
print("✓ Figure 1 saved to figure1_accuracy_curves.png")

# Generate Figure 2: Example signal space visualizations
print("\nGenerating Figure 2 (signal space examples)...")

fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
fig2.suptitle("Figure 2: Emergent Signal Space Structure", fontsize=14, fontweight='bold')

# Show signal space for main condition, low noise (best case)
key_main = ("main", 0.10)
if key_main in grouped:
    # Pick first run from this condition
    example_run = grouped[key_main][0]
    
    # We need to re-run to get the final producer state
    # For now, just add a note that this needs the actual signal data
    axes2[0].text(0.5, 0.5, "Low noise (σ=0.1)\nRun python experiment.py\nwith SAVE_FIGS=True\nto generate this panel", 
                 ha='center', va='center', fontsize=12)
    axes2[0].set_title("Main condition: σ=0.1 (final state)")
    
# Show signal space for high noise
key_high = ("main", 0.50)
if key_high in grouped:
    axes2[1].text(0.5, 0.5, "High noise (σ=0.5)\nRun python experiment.py\nwith SAVE_FIGS=True\nto generate this panel", 
                 ha='center', va='center', fontsize=12)
    axes2[1].set_title("Main condition: σ=0.5 (final state)")

for ax in axes2:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("signal dim 1")
    ax.set_ylabel("signal dim 2")

plt.tight_layout()
plt.savefig("figure2_signal_space_placeholder.png", dpi=300, bbox_inches='tight')
print("✓ Figure 2 placeholder saved (see instructions above)")

# Print summary statistics for the paper
print("\n" + "="*60)
print("SUMMARY STATISTICS FOR PAPER (generation 1000):")
print("="*60)

for key in sorted(grouped.keys()):
    condition, sigma = key
    runs = grouped[key]
    final_accs = [r['final_acc'] for r in runs]
    final_mis = [r['final_mi'] for r in runs]
    final_es = [r['final_effectiveness'] for r in runs]
    
    acc_mean = np.mean(final_accs)
    acc_std = np.std(final_accs, ddof=1)
    acc_ci = 1.96 * acc_std / np.sqrt(len(final_accs))
    
    mi_mean = np.mean(final_mis)
    e_mean = np.mean(final_es)
    
    print(f"\n{condition:15s} σ={sigma:.2f}:")
    print(f"  Accuracy: {acc_mean:.3f} ± {acc_ci:.3f}")
    print(f"  MI:       {mi_mean:.2f} bits")
    print(f"  E:        {e_mean:.2f}")

print("\n" + "="*60)
print("Copy these values into paper.md Table 1 and Results section!")
print("="*60)

plt.show()

