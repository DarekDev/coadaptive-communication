#!/usr/bin/env python3
"""
baldwinian_ablation.py -- isolates the contribution of Lamarckian inheritance.

The main experiment (experiment.py) is fundamentally LAMARCKIAN: offspring inherit the
parent's *learned* (end-of-life) weights. This ablation adds a genotype/phenotype split
to test the BALDWINIAN alternative, in which learning is NOT written back to the genome:

  - genotype = heritable initial weights.
  - At the start of each generation, phenotype <- genotype (learning is reset).
  - Within-lifetime learning modifies only the phenotype; fitness is measured on it.
  - Reproduction inherits the parent's GENOTYPE (+mutation); the learned phenotype is
    discarded. Selection can act only on how *learnable* a genotype is, never on what
    was learned.

Conditions (all at sigma=0.1):
  - lamarckian        : inherit learned weights (reproduces the paper's mechanism)
  - baldwinian        : inherit genotype only, 500 rounds/gen (same budget as main)
  - baldwinian_rich   : inherit genotype only, larger within-life interaction budget
                        (to test whether Baldwinian failure is merely a budget artefact)

Learning rules, selection, noise, and hyperparameters are copied EXACTLY from
experiment.py (eta=0.02, producer at 0.25*eta success-only, mutation N(0,0.02),
fitness-proportional resampling with replacement). Stdlib + numpy only.
"""
import json, time
import numpy as np

# --- constants (identical to experiment.py) ---
N = 100
N_CONTEXTS = 10
SIGNAL_DIM = 4
ETA = 0.02
PRODUCER_ETA = 0.25 * ETA
MUTATION_STD = 0.02
INIT_SCALE = 0.10
GENERATIONS = 1000
NOISE_SIGMA = 0.10


def new_pop(rng):
    """Each agent: dict with phenotype 'w', heritable 'geno', and 'fit'."""
    pop = []
    for _ in range(N):
        w = rng.normal(0.0, INIT_SCALE, size=(N_CONTEXTS, SIGNAL_DIM))
        pop.append({"w": w, "geno": w.copy(), "fit": 0.0})
    return pop


def estimate_accuracy(producers, receivers, rng, sigma, trials=1000):
    hit = 0
    for _ in range(trials):
        P = producers[rng.integers(0, N)]
        R = receivers[rng.integers(0, N)]
        c = int(rng.integers(0, N_CONTEXTS))
        s = P["w"][c]
        s_noisy = s + rng.normal(0.0, sigma, size=SIGNAL_DIM) if sigma > 0 else s
        guess = int(np.argmax(R["w"] @ s_noisy))
        hit += (guess == c)
    return hit / trials


def resample(pop, rng, mode):
    eps = 1e-9
    fits = np.array([max(a["fit"], 0.0) for a in pop], dtype=float)
    probs = (fits + eps) / max(fits.sum() + eps * len(pop), eps)
    out = []
    for _ in range(len(pop)):
        parent = pop[int(rng.choice(len(pop), p=probs))]
        if mode == "lamarckian":
            # inherit the LEARNED phenotype (+mutation)
            w = parent["w"] + rng.normal(0.0, MUTATION_STD, size=parent["w"].shape)
            out.append({"w": w, "geno": w.copy(), "fit": 0.0})
        else:
            # baldwinian: inherit the GENOTYPE (+mutation); discard learned phenotype
            geno = parent["geno"] + rng.normal(0.0, MUTATION_STD, size=parent["geno"].shape)
            out.append({"w": geno.copy(), "geno": geno, "fit": 0.0})
    return out


def run_seed(seed, mode, rounds_per_gen, generations=GENERATIONS, sigma=NOISE_SIGMA,
             curve_every=25, curve_trials=200, final_trials=2000):
    rng = np.random.default_rng(seed)
    producers, receivers = new_pop(rng), new_pop(rng)
    curve = []
    last_acc = 0.0
    for gen in range(generations):
        if mode != "lamarckian":
            # reset phenotype to genotype: learning does not persist across generations
            for a in producers: a["w"] = a["geno"].copy()
            for a in receivers: a["w"] = a["geno"].copy()
        for a in producers: a["fit"] = 0.0
        for a in receivers: a["fit"] = 0.0
        for _ in range(rounds_per_gen):
            P = producers[rng.integers(0, N)]
            R = receivers[rng.integers(0, N)]
            c = int(rng.integers(0, N_CONTEXTS))
            s = P["w"][c].copy()
            s_noisy = s + rng.normal(0.0, sigma, size=SIGNAL_DIM) if sigma > 0 else s
            guess = int(np.argmax(R["w"] @ s_noisy))
            reward = 1 if guess == c else 0
            # receiver update (exact copy of update_receiver)
            if guess == c:
                R["w"][c] += ETA * (s_noisy - R["w"][c])
            else:
                R["w"][guess] -= ETA * (s_noisy - R["w"][guess])
                R["w"][c]     += ETA * (s_noisy - R["w"][c])
            # producer update: success-only, slower (exact copy of update_producer call)
            if reward == 1:
                P["w"][c] += PRODUCER_ETA * (s_noisy - P["w"][c])
            P["fit"] += reward
            R["fit"] += reward
        # accuracy measured on the LEARNED phenotypes, before selection (as in experiment.py)
        if gen == generations - 1:
            last_acc = estimate_accuracy(producers, receivers, rng, sigma, trials=final_trials)
        if gen % curve_every == 0:
            curve.append((gen, estimate_accuracy(producers, receivers, rng, sigma, trials=curve_trials)))
        producers = resample(producers, rng, mode)
        receivers = resample(receivers, rng, mode)
    return last_acc, curve


def ci95(vals):
    a = np.array(vals, dtype=float)
    return float(a.mean()), float(1.96 * a.std(ddof=1) / np.sqrt(len(a)))


CONDITIONS = [
    ("lamarckian",      dict(mode="lamarckian", rounds_per_gen=500),  list(range(30))),
    ("baldwinian",      dict(mode="baldwinian", rounds_per_gen=500),  list(range(30))),
    ("baldwinian_rich", dict(mode="baldwinian", rounds_per_gen=2500), list(range(15))),
]

if __name__ == "__main__":
    t0 = time.time()
    results = {}
    for name, kw, seeds in CONDITIONS:
        accs, curves = [], {}
        for sd in seeds:
            acc, curve = run_seed(sd, **kw)
            accs.append(acc)
            curves[sd] = curve
        m, ci = ci95(accs)
        results[name] = {"rounds_per_gen": kw["rounds_per_gen"], "n_seeds": len(seeds),
                         "final_acc_mean": m, "final_acc_ci95": ci, "final_accs": accs}
        print(f"{name:16s} rounds={kw['rounds_per_gen']:5d} n={len(seeds):2d}  "
              f"final acc = {m:.3f} +/- {ci:.3f}   [{time.time()-t0:.0f}s elapsed]", flush=True)
    with open("baldwinian_ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDONE in {time.time()-t0:.0f}s -> baldwinian_ablation_results.json")
