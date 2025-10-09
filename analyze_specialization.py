#!/usr/bin/env python3
"""
Analyze emergent specialization in producer and receiver weights.
Computes quantitative metrics of how the two populations diverge over evolution.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

print("Loading results...")
with open("sweep_results_full.pkl", "rb") as f:
    all_results = pickle.load(f)

print(f"Loaded {len(all_results)} runs\n")


def compute_gini(weights):
    """
    Compute Gini coefficient for weight matrix.
    Measures inequality/sparsity: 0 = uniform, 1 = maximally sparse.
    """
    flat = np.abs(weights.flatten())
    flat = np.sort(flat)
    n = len(flat)
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * flat)) / (n * np.sum(flat)) - (n + 1) / n


def compute_l1_norm(weights):
    """Compute L1 norm (sum of absolute values)."""
    return np.sum(np.abs(weights))


def compute_l2_norm(weights):
    """Compute L2 (Frobenius) norm."""
    return np.linalg.norm(weights)


def compute_row_sparsity(weights):
    """Compute average sparsity per row (context)."""
    return np.mean(np.sum(np.abs(weights), axis=1))


def analyze_specialization(results, condition="main", noise_sigma=0.1, verbose=True):
    """
    Analyze specialization for a specific condition.
    
    Note: This function can't access final weights from the results dict,
    so it will re-run a simulation to get them. Alternatively, we could
    modify the main experiment to save final weights.
    """
    # For now, we'll create idealized analysis based on successful convergence
    # In a real implementation, you'd save final weights in sweep_results_full.pkl
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Analyzing: {condition}, σ={noise_sigma}")
        print(f"{'='*60}")
    
    # Filter results for this condition
    matching_runs = [r for r in results 
                     if r['condition'] == condition and r['noise_sigma'] == noise_sigma]
    
    if not matching_runs:
        print(f"No runs found for {condition}, σ={noise_sigma}")
        return None
    
    final_accs = [r['final_acc'] for r in matching_runs]
    avg_acc = np.mean(final_accs)
    
    if verbose:
        print(f"Found {len(matching_runs)} runs")
        print(f"Average final accuracy: {avg_acc:.3f}\n")
    
    # Since we can't extract final weights from the results,
    # we'll re-run ONE simulation to get them
    print("Re-running one simulation to extract final weights...")
    
    from experiment import run_sim
    result = run_sim(condition=condition, noise_sigma=noise_sigma, 
                     seed=42, show_plots=False, verbose=False)
    
    # Unfortunately run_sim doesn't return the final populations
    # We need to modify this to actually save and analyze weights
    # For now, let's document what SHOULD be computed
    
    return {
        'condition': condition,
        'noise_sigma': noise_sigma,
        'n_runs': len(matching_runs),
        'avg_accuracy': avg_acc,
    }


def run_single_with_weights(condition="main", noise_sigma=0.1, seed=42):
    """
    Run a single simulation and return final producer and receiver weights.
    This is a modified version that captures the final state.
    """
    import random
    from experiment import (
        N_PRODUCERS, N_RECEIVERS, N_CONTEXTS, SIGNAL_DIM, 
        ROUNDS_PER_GEN, GENERATIONS, LEARNING_RATE, MUTATION_STD,
        Producer, Receiver, sample_context, add_noise, receiver_guess,
        update_receiver, update_producer, ENABLE_PRODUCER_LEARNING
    )
    
    # Set seeds
    local_rng = np.random.default_rng(seed)
    random.seed(seed)
    
    def init_weights_local(rows, cols, scale=0.10):
        return local_rng.normal(0.0, scale, size=(rows, cols))
    
    # Initialize populations
    if condition == "shared":
        shared = init_weights_local(N_CONTEXTS, SIGNAL_DIM)
        producers = [Producer(shared.copy()) for _ in range(N_PRODUCERS)]
        receivers = [Receiver(shared.copy()) for _ in range(N_RECEIVERS)]
    else:
        producers = [Producer(init_weights_local(N_CONTEXTS, SIGNAL_DIM)) for _ in range(N_PRODUCERS)]
        receivers = [Receiver(init_weights_local(N_CONTEXTS, SIGNAL_DIM)) for _ in range(N_RECEIVERS)]
    
    if condition == "fixed_mapping":
        frozen_producer_weights = [p.weights.copy() for p in producers]
    
    # Run evolution
    for gen in range(1, GENERATIONS + 1):
        for p in producers: p.fitness = 0.0
        for r in receivers: r.fitness = 0.0
        
        for _ in range(ROUNDS_PER_GEN):
            P = random.choice(producers)
            R = random.choice(receivers)
            c = int(local_rng.integers(0, N_CONTEXTS))
            
            if condition == "no_comm":
                guess = int(local_rng.integers(0, N_CONTEXTS))
                reward = 1 if guess == c else 0
            else:
                s = P.weights[c].copy()
                s_noisy = s + local_rng.normal(0.0, noise_sigma, size=s.shape) if noise_sigma > 0 else s
                guess = receiver_guess(R.weights, s_noisy)
                reward = 1 if guess == c else 0
                
                update_receiver(R.weights, s_noisy, c, guess, LEARNING_RATE, 0.0)
                
                if ENABLE_PRODUCER_LEARNING and condition != "fixed_mapping" and reward == 1:
                    update_producer(P.weights, s_noisy, c, LEARNING_RATE * 0.25)
            
            P.fitness += reward
            R.fitness += reward
        
        # Evolve
        if condition == "fixed_mapping":
            for i, p in enumerate(producers):
                p.weights = frozen_producer_weights[i].copy()
                p.fitness = 0.0
        else:
            producers = resample_local(producers, local_rng)
        
        receivers = resample_local(receivers, local_rng)
    
    return producers, receivers


def resample_local(pop, rng_local, mutation_std=0.02):
    """Local copy of resampling function."""
    import copy
    from experiment import Producer, Receiver
    
    eps = 1e-9
    fitnesses = np.array([max(p.fitness, 0.0) for p in pop], dtype=float)
    probs = (fitnesses + eps) / max(fitnesses.sum() + eps * len(pop), eps)
    
    new_pop = []
    for _ in range(len(pop)):
        parent_idx = rng_local.choice(len(pop), p=probs)
        parent = pop[parent_idx]
        child = copy.deepcopy(parent)
        if isinstance(child, Producer) or isinstance(child, Receiver):
            child.weights = child.weights + rng_local.normal(0.0, mutation_std, size=child.weights.shape)
            child.fitness = 0.0
        new_pop.append(child)
    return new_pop


# Main analysis
print("\n" + "="*60)
print("WEIGHT SPECIALIZATION ANALYSIS")
print("="*60)

# Run fresh simulation to get final weights
conditions_to_analyze = [
    ("main", 0.1),
    ("main", 0.5),
    ("shared", 0.1),
]

results_summary = []

for condition, sigma in conditions_to_analyze:
    print(f"\n>>> Running {condition}, σ={sigma}...")
    
    producers, receivers = run_single_with_weights(condition=condition, noise_sigma=sigma, seed=42)
    
    # Aggregate statistics across population
    producer_weights = np.array([p.weights for p in producers])  # (n_producers, n_contexts, signal_dim)
    receiver_weights = np.array([r.weights for r in receivers])  # (n_receivers, n_contexts, signal_dim)
    
    # Compute average weight matrices
    avg_producer = producer_weights.mean(axis=0)  # (n_contexts, signal_dim)
    avg_receiver = receiver_weights.mean(axis=0)  # (n_contexts, signal_dim)
    
    # Metrics
    metrics = {
        'condition': condition,
        'noise_sigma': sigma,
        'producer_gini': compute_gini(avg_producer),
        'receiver_gini': compute_gini(avg_receiver),
        'producer_l1': compute_l1_norm(avg_producer),
        'receiver_l1': compute_l1_norm(avg_receiver),
        'producer_l2': compute_l2_norm(avg_producer),
        'receiver_l2': compute_l2_norm(avg_receiver),
        'weight_cosine_similarity': np.mean([
            np.dot(avg_producer[i], avg_receiver[i]) / 
            (np.linalg.norm(avg_producer[i]) * np.linalg.norm(avg_receiver[i]) + 1e-12)
            for i in range(len(avg_producer))
        ]),
    }
    
    results_summary.append(metrics)
    
    print(f"\nProducer weights:")
    print(f"  Gini coefficient: {metrics['producer_gini']:.3f}")
    print(f"  L1 norm:          {metrics['producer_l1']:.2f}")
    print(f"  L2 norm:          {metrics['producer_l2']:.2f}")
    
    print(f"\nReceiver weights:")
    print(f"  Gini coefficient: {metrics['receiver_gini']:.3f}")
    print(f"  L1 norm:          {metrics['receiver_l1']:.2f}")
    print(f"  L2 norm:          {metrics['receiver_l2']:.2f}")
    
    print(f"\nCross-population:")
    print(f"  Cosine similarity (avg): {metrics['weight_cosine_similarity']:.3f}")

# Summary table
print("\n" + "="*60)
print("SUMMARY TABLE")
print("="*60)
print(f"{'Condition':<15} {'σ':<6} {'Prod Gini':<12} {'Recv Gini':<12} {'Cosine Sim':<12}")
print("-"*60)
for m in results_summary:
    print(f"{m['condition']:<15} {m['noise_sigma']:<6.2f} {m['producer_gini']:<12.3f} "
          f"{m['receiver_gini']:<12.3f} {m['weight_cosine_similarity']:<12.3f}")

# Generate visualization
print("\nGenerating weight heatmaps...")
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle("Weight Matrix Specialization", fontsize=14, fontweight='bold')

for idx, (condition, sigma) in enumerate(conditions_to_analyze):
    producers, receivers = run_single_with_weights(condition=condition, noise_sigma=sigma, seed=42)
    
    avg_producer = np.array([p.weights for p in producers]).mean(axis=0)
    avg_receiver = np.array([r.weights for r in receivers]).mean(axis=0)
    
    # Producer heatmap
    ax_p = axes[0, idx]
    im_p = ax_p.imshow(avg_producer, aspect='auto', cmap='RdBu_r', vmin=-2, vmax=2)
    ax_p.set_title(f"{condition} σ={sigma}\nProducer")
    ax_p.set_xlabel("Signal dim")
    ax_p.set_ylabel("Context")
    plt.colorbar(im_p, ax=ax_p, fraction=0.046)
    
    # Receiver heatmap
    ax_r = axes[1, idx]
    im_r = ax_r.imshow(avg_receiver, aspect='auto', cmap='RdBu_r', vmin=-2, vmax=2)
    ax_r.set_title(f"Receiver")
    ax_r.set_xlabel("Signal dim")
    ax_r.set_ylabel("Context")
    plt.colorbar(im_r, ax=ax_r, fraction=0.046)

plt.tight_layout()
plt.savefig("figure3_specialization_heatmaps.png", dpi=300, bbox_inches='tight')
print("✓ Saved figure3_specialization_heatmaps.png")

print("\n" + "="*60)
print("Analysis complete!")
print("="*60)

