#!/usr/bin/env python3
"""
Co-adaptive communication evolution between producers and receivers.
Accompanying code for: "Co-adaptation of Signalling and Perception in Multi-Agent Systems"
"""

import math
import random
import copy
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import track
    from rich.text import Text
except ImportError as e:
    raise SystemExit(
        "Missing dependencies. Install with:\n"
        "    pip install numpy matplotlib rich\n"
    )

console = Console()

# ==============================================================================
# PARAMETERS
# ==============================================================================

SEED = 42                 # RNG seed for reproducibility
N_PRODUCERS = 100         # producers in population
N_RECEIVERS = 100         # receivers in population
N_CONTEXTS = 10           # number of distinct contexts (0..N_CONTEXTS-1)
SIGNAL_DIM = 4            # 4D signals (optimized via grid search)
ROUNDS_PER_GEN = 500      # interactions per generation
GENERATIONS = 1000        # total generations (ensures full convergence across all noise conditions)
NOISE_SIGMA = 0.10        # channel noise std dev for signal
LEARNING_RATE = 0.02      # eta for weight nudges (optimized via grid search)
MUTATION_STD = 0.02       # std dev for evolutionary weight mutation (optimized via grid search)
DECAY_ON_FAIL = 0.0       # optional tiny weight decay on failure (0.0 ok)
ENABLE_PRODUCER_LEARNING = True   # symmetric learning: both producers and receivers adapt within lifetime

# Experimental condition: "main" | "no_comm" | "fixed_mapping" | "shared"
CONDITION = "main"

# Plotting
DARK_BG = True
PLOT_LIVE = False
SAVE_FIGS = False

# ==============================================================================
# MODEL DEFINITIONS
# ==============================================================================

rng = np.random.default_rng(SEED)
random.seed(SEED)

def init_weights(rows: int, cols: int, scale: float = 0.10) -> np.ndarray:
    return rng.normal(0.0, scale, size=(rows, cols))

@dataclass
class Producer:
    weights: np.ndarray  # shape (N_CONTEXTS, SIGNAL_DIM)
    fitness: float = 0.0

@dataclass
class Receiver:
    weights: np.ndarray  # shape (N_CONTEXTS, SIGNAL_DIM)
    fitness: float = 0.0

def sample_context(n_contexts: int) -> int:
    return rng.integers(0, n_contexts)

def add_noise(signal: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return signal
    return signal + rng.normal(0.0, sigma, size=signal.shape)

def receiver_guess(receiver_w: np.ndarray, signal_noisy: np.ndarray) -> int:
    """Decode signal by computing scores for each context and taking argmax."""
    scores = receiver_w @ signal_noisy
    return int(np.argmax(scores))

def update_receiver(receiver_w: np.ndarray, signal_noisy: np.ndarray, true_c: int, guess: int, eta: float, decay: float = 0.0):
    """Hebbian-like weight update: reinforce correct associations, weaken incorrect ones."""
    if true_c == guess:
        receiver_w[true_c] += eta * (signal_noisy - receiver_w[true_c])
    else:
        receiver_w[guess]   -= eta * (signal_noisy - receiver_w[guess])
        receiver_w[true_c]  += eta * (signal_noisy - receiver_w[true_c])
        if decay > 0.0:
            receiver_w *= (1.0 - decay)

def update_producer(producer_w: np.ndarray, signal_emitted: np.ndarray, true_c: int, eta: float):
    """Stabilize producer mapping toward successful signals."""
    producer_w[true_c] += eta * (signal_emitted - producer_w[true_c])

def fitness_weighted_resample(pop: List, mutation_std: float) -> List:
    """Fitness-proportional selection with replacement and Gaussian mutation."""
    eps = 1e-9
    fitnesses = np.array([max(p.fitness, 0.0) for p in pop], dtype=float)
    probs = (fitnesses + eps) / max(fitnesses.sum() + eps * len(pop), eps)

    new_pop = []
    for _ in range(len(pop)):
        parent_idx = rng.choice(len(pop), p=probs)
        parent = pop[parent_idx]
        child = copy.deepcopy(parent)
        # mutate weights
        if isinstance(child, Producer) or isinstance(child, Receiver):
            child.weights = child.weights + rng.normal(0.0, mutation_std, size=child.weights.shape)
            child.fitness = 0.0
        new_pop.append(child)
    return new_pop

def estimate_accuracy(producers: List[Producer], receivers: List[Receiver], trials: int = 200) -> float:
    """Estimate current accuracy without learning side-effects."""
    correct = 0
    for _ in range(trials):
        P = random.choice(producers)
        R = random.choice(receivers)
        c = sample_context(N_CONTEXTS)
        signal = P.weights[c].copy()
        signal_noisy = add_noise(signal, NOISE_SIGMA)
        guess = receiver_guess(R.weights, signal_noisy)
        correct += int(guess == c)
    return correct / float(trials)

def estimate_mutual_information(producers: List[Producer], receivers: List[Receiver], trials:int=2000, bins:int=16) -> Tuple[float, float, float]:
    """
    Crude MI estimate between contexts C and binned 2D messages M.
    We bin the 2D signal into bins x bins grid and compute I(C; M_bin).
    
    Returns:
        mi: mutual information I(C;M) in bits
        h_m: signal entropy H(M) in bits
        effectiveness: E = I(C;M) / H(C), normalized MI in [0,1]
    """
    # collect samples without learning
    Cs = []
    Ms = []
    for _ in range(trials):
        P = random.choice(producers)
        R = random.choice(receivers)  # receiver not used for MI; we only need signals from producers
        c = sample_context(N_CONTEXTS)
        s = add_noise(P.weights[c].copy(), NOISE_SIGMA)
        Cs.append(c)
        Ms.append(s)
    Cs = np.array(Cs, dtype=int)
    Ms = np.array(Ms, dtype=float)

    # bin 2D signals to a single index
    # scale to [0,1] per dim then to bins
    mins = Ms.min(axis=0)
    maxs = Ms.max(axis=0)
    spans = np.maximum(maxs - mins, 1e-9)
    norm = (Ms - mins) / spans
    xb = np.clip((norm[:,0] * bins).astype(int), 0, bins-1)
    yb = np.clip((norm[:,1] * bins).astype(int), 0, bins-1)
    mb = xb * bins + yb  # single bin index in 0..bins*bins-1

    Kc = N_CONTEXTS
    Km = bins * bins
    # joint counts
    joint = np.zeros((Kc, Km), dtype=float)
    for c, m in zip(Cs, mb):
        joint[c, m] += 1.0
    joint /= joint.sum()

    pc = joint.sum(axis=1, keepdims=True)  # (Kc,1)
    pm = joint.sum(axis=0, keepdims=True)  # (1,Km)

    # I(C;M) = sum p(c,m) log [ p(c,m) / (p(c)p(m)) ]
    eps = 1e-12
    ratio = (joint + eps) / (pc @ pm + eps)
    mi = float(np.sum(joint * np.log2(ratio)))
    
    # H(M) = signal entropy
    h_m = float(-(pm * np.log2(pm + eps)).sum())
    
    # H(C) = context entropy (uniform distribution over N_CONTEXTS)
    h_c = math.log2(N_CONTEXTS)
    
    # Effectiveness: normalized reduction in context uncertainty
    effectiveness = mi / h_c
    
    return mi, h_m, effectiveness

# ==============================================================================
# SIMULATION
# ==============================================================================

def run_sim(condition: str = None, noise_sigma: float = None, seed: int = None, show_plots: bool = True, verbose: bool = True):
    """
    Run a single simulation.
    
    Args:
        condition: "main" | "no_comm" | "fixed_mapping" | "shared" (defaults to global CONDITION)
        noise_sigma: channel noise std dev (defaults to global NOISE_SIGMA)
        seed: RNG seed (defaults to global SEED)
        show_plots: whether to display plots at the end
        verbose: whether to print progress to console
    
    Returns:
        dict with keys: 'acc_history', 'mi_history', 'h_m_history', 'effectiveness_history',
                        'final_acc', 'final_mi', 'final_h_m', 'final_effectiveness'
    """
    # Use provided params or fall back to globals
    condition = condition if condition is not None else CONDITION
    noise_sigma = noise_sigma if noise_sigma is not None else NOISE_SIGMA
    seed = seed if seed is not None else SEED
    
    # Set seeds for this run
    local_rng = np.random.default_rng(seed)
    random.seed(seed)
    
    # Theme for plots
    if DARK_BG:
        plt.style.use("dark_background")

    # Helper to use local_rng for initialization
    def init_weights_local(rows: int, cols: int, scale: float = 0.10) -> np.ndarray:
        return local_rng.normal(0.0, scale, size=(rows, cols))
    
    # init populations based on condition
    if condition == "shared":
        # Upper bound: share a single table both ways
        shared = init_weights_local(N_CONTEXTS, SIGNAL_DIM)
        producers = [Producer(shared.copy()) for _ in range(N_PRODUCERS)]
        receivers = [Receiver(shared.copy()) for _ in range(N_RECEIVERS)]
    else:
        # Standard initialization: independent populations
        producers = [Producer(init_weights_local(N_CONTEXTS, SIGNAL_DIM)) for _ in range(N_PRODUCERS)]
        receivers = [Receiver(init_weights_local(N_CONTEXTS, SIGNAL_DIM)) for _ in range(N_RECEIVERS)]
    
    # For fixed_mapping: freeze producer weights (never mutate, never learn)
    if condition == "fixed_mapping":
        frozen_producer_weights = [p.weights.copy() for p in producers]

    acc_history = []
    mi_history = []
    h_m_history = []
    effectiveness_history = []

    # Progress header
    if verbose:
        condition_label = {
            "main": "main experiment",
            "no_comm": "no communication control",
            "fixed_mapping": "fixed mapping control",
            "shared": "shared weights control"
        }.get(condition, condition)
        
        banner = Panel(
            Text("Co-Adaptive Communication Evolution", style="bold cyan"),
            subtitle=f"Condition: {condition_label} | Seed: {seed}",
            border_style="cyan",
        )
        console.print(banner)

    for gen in range(1, GENERATIONS + 1):
        # reset fitness each generation
        for p in producers: p.fitness = 0.0
        for r in receivers: r.fitness = 0.0

        # interactions
        for _ in range(ROUNDS_PER_GEN):
            P = random.choice(producers)
            R = random.choice(receivers)
            c = sample_context(N_CONTEXTS)

            # Handle different conditions
            if condition == "no_comm":
                # No communication: receiver guesses randomly (ignores signal)
                guess = local_rng.integers(0, N_CONTEXTS)
                reward = 1 if guess == c else 0
            else:
                s = P.weights[c].copy()                        # emit
                s_noisy = s + local_rng.normal(0.0, noise_sigma, size=s.shape) if noise_sigma > 0 else s
                guess = receiver_guess(R.weights, s_noisy)     # decode
                reward = 1 if guess == c else 0

                # learning (within lifetime)
                update_receiver(R.weights, s_noisy, c, guess, LEARNING_RATE, DECAY_ON_FAIL)
                
                # Producer learning (disabled for fixed_mapping condition)
                if ENABLE_PRODUCER_LEARNING and condition != "fixed_mapping" and reward == 1:
                    update_producer(P.weights, s_noisy, c, LEARNING_RATE * 0.25)  # slower, more stable

            # fitness
            P.fitness += reward
            R.fitness += reward

        # metrics (probe without learning side-effects)
        acc = estimate_accuracy_local(producers, receivers, local_rng, noise_sigma, trials=400)
        mi, h_m, effectiveness = estimate_mi_local(producers, receivers, local_rng, noise_sigma, trials=2000, bins=16)
        acc_history.append(acc)
        mi_history.append(mi)
        h_m_history.append(h_m)
        effectiveness_history.append(effectiveness)

        # retro console line
        if verbose:
            color = "bright_green" if acc > 0.8 else ("yellow" if acc > 0.5 else "red")
            console.print(
                f"[steel_blue]GEN {gen:03d}[/]  "
                f"[{color}]accuracy={acc:0.3f}[/]  "
                f"[magenta]MI={mi:0.2f} bits[/]  "
                f"[cyan]H(M)={h_m:0.2f}[/]  "
                f"[green]E={effectiveness:0.2f}[/]  "
                f"[grey50]σ={noise_sigma:0.2f} η={LEARNING_RATE:0.3f}[/]"
            )

        # evolve (between generations)
        if condition == "fixed_mapping":
            # Fixed mapping: producers never evolve, restore frozen weights
            for i, p in enumerate(producers):
                p.weights = frozen_producer_weights[i].copy()
                p.fitness = 0.0
        else:
            # Normal evolution for producers
            producers = resample_local(producers, local_rng, mutation_std=MUTATION_STD)
        
        # Receivers always evolve (except in no_comm they have no useful signal to learn from)
        receivers = resample_local(receivers, local_rng, mutation_std=MUTATION_STD)

    # Return results
    results = {
        'acc_history': acc_history,
        'mi_history': mi_history,
        'h_m_history': h_m_history,
        'effectiveness_history': effectiveness_history,
        'final_acc': acc_history[-1] if acc_history else 0.0,
        'final_mi': mi_history[-1] if mi_history else 0.0,
        'final_h_m': h_m_history[-1] if h_m_history else 0.0,
        'final_effectiveness': effectiveness_history[-1] if effectiveness_history else 0.0,
        'condition': condition,
        'noise_sigma': noise_sigma,
        'seed': seed
    }
    
    # ============ plotting ============
    if show_plots:
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))

        # Left: accuracy curve
        ax[0].plot(np.arange(1, GENERATIONS+1), acc_history, lw=2)
        ax[0].set_title(f"Alignment Accuracy over Generations\n{condition}, σ={noise_sigma}")
        ax[0].set_xlabel("Generation")
        ax[0].set_ylabel("Accuracy (probe)")

        # Right: final signal space scatter (color by context)
        # build signals for one representative producer (averaging across pop is noisy)
        P0 = random.choice(producers)
        # generate multiple samples per context to show noise cloud
        samples = []
        labels = []
        reps_per_context = 40
        for c in range(N_CONTEXTS):
            base = P0.weights[c]
            for _ in range(reps_per_context):
                noisy = base + local_rng.normal(0.0, noise_sigma, size=base.shape) if noise_sigma > 0 else base
                samples.append(noisy)
                labels.append(c)
        samples = np.array(samples)
        labels = np.array(labels)
        sc = ax[1].scatter(samples[:, 0], samples[:, 1], c=labels, cmap="tab10", s=18, alpha=0.85, edgecolors="none")
        title_suffix = " (2D projection)" if samples.shape[1] > 2 else ""
        ax[1].set_title(f"Final Signal Space (one producer){title_suffix}")
        ax[1].set_xlabel("signal dim 1")
        ax[1].set_ylabel("signal dim 2")

        plt.tight_layout()
        if SAVE_FIGS:
            plt.savefig(f"retro_comm_figs_{condition}_sigma{noise_sigma:.2f}_seed{seed}.png", dpi=180)
        plt.show()
    
    return results


# ==============================================================================
# LOCAL RNG HELPERS (for parallel-safe execution)
# ==============================================================================

def sample_context_local(rng_local, n_contexts: int) -> int:
    return rng_local.integers(0, n_contexts)

def estimate_accuracy_local(producers, receivers, rng_local, noise_sigma, trials: int = 200) -> float:
    correct = 0
    for _ in range(trials):
        P = random.choice(producers)
        R = random.choice(receivers)
        c = sample_context_local(rng_local, N_CONTEXTS)
        signal = P.weights[c].copy()
        signal_noisy = signal + rng_local.normal(0.0, noise_sigma, size=signal.shape) if noise_sigma > 0 else signal
        guess = receiver_guess(R.weights, signal_noisy)
        correct += int(guess == c)
    return correct / float(trials)

def estimate_mi_local(producers, receivers, rng_local, noise_sigma, trials: int = 2000, bins: int = 16):
    Cs = []
    Ms = []
    for _ in range(trials):
        P = random.choice(producers)
        c = sample_context_local(rng_local, N_CONTEXTS)
        s = P.weights[c].copy()
        s_noisy = s + rng_local.normal(0.0, noise_sigma, size=s.shape) if noise_sigma > 0 else s
        Cs.append(c)
        Ms.append(s_noisy)
    Cs = np.array(Cs, dtype=int)
    Ms = np.array(Ms, dtype=float)

    mins = Ms.min(axis=0)
    maxs = Ms.max(axis=0)
    spans = np.maximum(maxs - mins, 1e-9)
    norm = (Ms - mins) / spans
    xb = np.clip((norm[:,0] * bins).astype(int), 0, bins-1)
    yb = np.clip((norm[:,1] * bins).astype(int), 0, bins-1)
    mb = xb * bins + yb

    Kc = N_CONTEXTS
    Km = bins * bins
    joint = np.zeros((Kc, Km), dtype=float)
    for c, m in zip(Cs, mb):
        joint[c, m] += 1.0
    joint /= joint.sum()

    pc = joint.sum(axis=1, keepdims=True)
    pm = joint.sum(axis=0, keepdims=True)

    eps = 1e-12
    ratio = (joint + eps) / (pc @ pm + eps)
    mi = float(np.sum(joint * np.log2(ratio)))
    h_m = float(-(pm * np.log2(pm + eps)).sum())
    h_c = math.log2(N_CONTEXTS)
    effectiveness = mi / h_c
    
    return mi, h_m, effectiveness

def resample_local(pop, rng_local, mutation_std: float):
    """
    Fitness-proportional selection with replacement.
    Adds small epsilon to avoid divide-by-zero when all fitnesses are zero.
    """
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


# ==============================================================================
# PARAMETER SWEEP
# ==============================================================================

def _run_single_config(args):
    """Helper function for parallel execution."""
    cond, sigma, seed = args
    import time
    start = time.time()
    result = run_sim(condition=cond, noise_sigma=sigma, seed=seed, 
                    show_plots=False, verbose=False)
    elapsed = time.time() - start
    return result, elapsed


def run_sweep(noise_grid=None, conditions=None, n_seeds=30, save_results=True, parallel=False, n_workers=None):
    """
    Run parameter sweep over noise levels, conditions, and seeds.
    
    Args:
        noise_grid: list of noise sigmas to test (default: [0.10, 0.50, 0.60])
        conditions: list of conditions to test (default: all 4)
        n_seeds: number of independent seeds per config (default: 30)
        save_results: whether to save aggregated results to disk
        parallel: whether to use multiprocessing (default: False)
        n_workers: number of parallel workers (default: CPU count - 1)
    
    Returns:
        dict with aggregated results per (condition, noise) combo
    """
    if noise_grid is None:
        noise_grid = [0.10, 0.50, 0.60]
    if conditions is None:
        conditions = ["main", "no_comm", "fixed_mapping", "shared"]
    
    seeds = [100 + i for i in range(n_seeds)]
    
    all_results = []
    total_runs = len(conditions) * len(noise_grid) * n_seeds
    
    console.print(f"\n[bold cyan]PARAMETER SWEEP[/]")
    console.print(f"Conditions: {conditions}")
    console.print(f"Noise levels: {noise_grid}")
    console.print(f"Seeds: {n_seeds}")
    console.print(f"Total runs: {total_runs}")
    
    if parallel:
        import multiprocessing as mp
        if n_workers is None:
            n_workers = max(1, mp.cpu_count() - 1)
        console.print(f"Parallel mode: {n_workers} workers\n")
        
        # Build list of all configs
        configs = []
        for cond in conditions:
            for sigma in noise_grid:
                for s in seeds:
                    configs.append((cond, sigma, s))
        
        # Run in parallel with progress bar
        import time
        start_time = time.time()
        completed = 0
        
        with mp.Pool(n_workers) as pool:
            for result, elapsed in pool.imap_unordered(_run_single_config, configs):
                completed += 1
                all_results.append(result)
                
                # Simple progress
                elapsed_total = time.time() - start_time
                avg_time = elapsed_total / completed
                eta_minutes = (total_runs - completed) * avg_time / 60
                
                console.print(
                    f"  [{completed}/{total_runs}] "
                    f"{result['condition']:15s} σ={result['noise_sigma']:.2f} seed={result['seed']:3d} → "
                    f"acc={result['final_acc']:.3f} | "
                    f"ETA: {eta_minutes:.1f}min"
                )
    else:
        console.print(f"Sequential mode\n")
        
        run_count = 0
        import time
        run_times = []
        
        for cond in conditions:
            for sigma in noise_grid:
                console.print(f"\n[yellow]>>> Running {cond} with σ={sigma}...[/]")
                for s in seeds:
                    run_count += 1
                    
                    # Time this run
                    start_time = time.time()
                    result = run_sim(condition=cond, noise_sigma=sigma, seed=s, 
                                   show_plots=False, verbose=False)
                    elapsed = time.time() - start_time
                    run_times.append(elapsed)
                    
                    all_results.append(result)
                    
                    # Calculate mean time of last 10 runs
                    recent_times = run_times[-10:]
                    mean_time = sum(recent_times) / len(recent_times)
                    remaining = total_runs - run_count
                    eta_minutes = (remaining * mean_time) / 60
                    
                    console.print(
                        f"  [{run_count}/{total_runs}] seed={s:3d} → "
                        f"acc={result['final_acc']:.3f} | "
                        f"{elapsed:.1f}s | "
                        f"ETA: {eta_minutes:.1f}min"
                    )
    
    # Aggregate results
    aggregated = aggregate_results(all_results)
    
    if save_results:
        import json
        import csv
        import pickle
        
        # Save aggregated results to JSON
        with open("sweep_results.json", "w") as f:
            export = {}
            for key, val in aggregated.items():
                export[str(key)] = {k: (v.tolist() if isinstance(v, np.ndarray) else v) 
                                   for k, v in val.items()}
            json.dump(export, f, indent=2)
        console.print(f"[green]✓ Aggregated results saved to sweep_results.json[/]")
        
        # Save aggregated results to CSV for easy plotting
        with open("sweep_results.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["condition", "noise_sigma", "n_runs", 
                           "acc_mean", "acc_ci", "mi_mean", "mi_ci", "e_mean", "e_ci"])
            for key in sorted(aggregated.keys()):
                agg = aggregated[key]
                writer.writerow([
                    agg['condition'], agg['noise_sigma'], agg['n_runs'],
                    agg['acc_mean'], agg['acc_ci'],
                    agg['mi_mean'], agg['mi_ci'],
                    agg['e_mean'], agg['e_ci']
                ])
        console.print(f"[green]✓ Aggregated results saved to sweep_results.csv[/]")
        
        # Save full raw results (including histories) for plotting trajectories
        with open("sweep_results_full.pkl", "wb") as f:
            pickle.dump(all_results, f)
        console.print(f"[green]✓ Full results (with histories) saved to sweep_results_full.pkl[/]")
    
    return aggregated


def aggregate_results(all_results):
    """
    Aggregate results by (condition, noise_sigma) and compute mean ± 95% CI.
    Uses standard error: 1.96 * (sd / sqrt(n)) for 95% confidence.
    """
    from collections import defaultdict
    grouped = defaultdict(list)
    
    for r in all_results:
        key = (r['condition'], r['noise_sigma'])
        grouped[key].append(r)
    
    aggregated = {}
    for key, runs in grouped.items():
        final_accs = [r['final_acc'] for r in runs]
        final_mis = [r['final_mi'] for r in runs]
        final_es = [r['final_effectiveness'] for r in runs]
        
        aggregated[key] = {
            'condition': key[0],
            'noise_sigma': key[1],
            'n_runs': len(runs),
            'acc_mean': np.mean(final_accs),
            'acc_ci': compute_ci_95(final_accs),
            'mi_mean': np.mean(final_mis),
            'mi_ci': compute_ci_95(final_mis),
            'e_mean': np.mean(final_es),
            'e_ci': compute_ci_95(final_es),
        }
    
    return aggregated


def compute_ci_95(data):
    """Compute 95% confidence interval using standard error: 1.96 * (sd / sqrt(n))."""
    data = np.array(data)
    n = len(data)
    if n < 2:
        return 0.0
    std = np.std(data, ddof=1)  # sample standard deviation
    se = std / np.sqrt(n)        # standard error
    ci_95 = 1.96 * se            # 95% CI half-width
    return ci_95


def print_table(aggregated):
    """Print results in a nice table format."""
    table = Table(title="Aggregated Results (Final Accuracy)")
    table.add_column("Condition", style="cyan")
    table.add_column("Noise σ", style="magenta")
    table.add_column("Accuracy (mean ± 95% CI)", style="green")
    
    for key in sorted(aggregated.keys()):
        agg = aggregated[key]
        table.add_row(
            agg['condition'],
            f"{agg['noise_sigma']:.2f}",
            f"{agg['acc_mean']:.3f} ± {agg['acc_ci']:.3f}"
        )
    
    console.print(table)


if __name__ == "__main__":
    # Full parameter sweep (reproduces paper results)
    # Uses multiprocessing for ~8-10x speedup
    # Runs: 4 conditions × 3 noise levels × 30 seeds = 360 experiments
    results = run_sweep(parallel=True)
    print_table(results)
    
    # For single test run, uncomment:
    # result = run_sim(condition="main", noise_sigma=0.1, seed=42, show_plots=True, verbose=True)