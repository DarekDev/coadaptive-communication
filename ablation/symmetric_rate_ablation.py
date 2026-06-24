#!/usr/bin/env python3
"""
symmetric_rate_ablation.py -- tests whether the producer/receiver LEARNING-RATE
asymmetry is load-bearing for convergence.

Reviewer concern: the main model gives producers a deliberately weakened update
rule -- they learn at one quarter of the base rate (0.25*eta) and ONLY on success
-- while receivers learn at the full rate eta on every interaction (hit and miss).
This was chosen because it "empirically stabilized convergence." A reviewer may ask
whether the result depends on that hand-tuned asymmetry.

This script isolates that choice. It is an ADDITIVE companion to
baldwinian_ablation.py -- it does not modify experiment.py, the repository, or any
existing setup. The original asymmetric rule remains the default of the main model;
here it is simply one of two options so the two can be compared head to head.

producer_rule:
  - "asymmetric" : the PUBLISHED rule. Producer updates only on success (reward==1),
                   at 0.25*eta, nudging the emitted context row toward the received
                   (noisy) signal. (Exact copy of experiment.py / baldwinian_ablation.py.)
  - "symmetric"  : the BASE rule. Producer updates on EVERY interaction (hit or miss),
                   at the full base rate eta, nudging the emitted context row toward
                   the received (noisy) signal. (Removes both asymmetries: the 0.25
                   slowdown and the success-only restriction.)

Everything else -- receiver rule, noise, mutation, fitness-proportional resampling,
genotype/phenotype split for the Baldwinian arm, hyperparameters -- is IDENTICAL to
baldwinian_ablation.py. Conditions are run at sigma=0.1, 500 rounds/gen, 30 seeds,
crossing {asymmetric, symmetric} x {lamarckian, baldwinian}. The asymmetric arms
reproduce the published Table 2 values (consistency check). Stdlib + numpy only.
"""
import json, time
import numpy as np

# --- constants (identical to experiment.py / baldwinian_ablation.py) ---
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
    pop = []
    for _ in range(N):
        w = rng.normal(0.0, INIT_SCALE, size=(N_CONTEXTS, SIGNAL_DIM))
        pop.append({"w": w, "geno": w.copy(), "fit": 0.0})
    return pop


def estimate_accuracy(producers, receivers, rng, sigma, trials=2000):
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
            w = parent["w"] + rng.normal(0.0, MUTATION_STD, size=parent["w"].shape)
            out.append({"w": w, "geno": w.copy(), "fit": 0.0})
        else:
            geno = parent["geno"] + rng.normal(0.0, MUTATION_STD, size=parent["geno"].shape)
            out.append({"w": geno.copy(), "geno": geno, "fit": 0.0})
    return out


def run_seed(seed, mode, producer_rule, rounds_per_gen=500, generations=GENERATIONS,
             sigma=NOISE_SIGMA, final_trials=2000):
    rng = np.random.default_rng(seed)
    producers, receivers = new_pop(rng), new_pop(rng)
    last_acc = 0.0
    for gen in range(generations):
        if mode != "lamarckian":
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
            # receiver update (unchanged, full eta, hit+miss)
            if guess == c:
                R["w"][c] += ETA * (s_noisy - R["w"][c])
            else:
                R["w"][guess] -= ETA * (s_noisy - R["w"][guess])
                R["w"][c]     += ETA * (s_noisy - R["w"][c])
            # producer update
            if producer_rule == "asymmetric":
                if reward == 1:
                    P["w"][c] += PRODUCER_ETA * (s_noisy - P["w"][c])
            else:  # symmetric: full eta, every interaction
                P["w"][c] += ETA * (s_noisy - P["w"][c])
            P["fit"] += reward
            R["fit"] += reward
        if gen == generations - 1:
            last_acc = estimate_accuracy(producers, receivers, rng, sigma, trials=final_trials)
        producers = resample(producers, rng, mode)
        receivers = resample(receivers, rng, mode)
    return last_acc


def ci95(vals):
    a = np.array(vals, dtype=float)
    return float(a.mean()), float(1.96 * a.std(ddof=1) / np.sqrt(len(a)))


CONDITIONS = [
    ("asymmetric_lamarckian", dict(mode="lamarckian", producer_rule="asymmetric"), list(range(30))),
    ("asymmetric_baldwinian", dict(mode="baldwinian", producer_rule="asymmetric"), list(range(30))),
    ("symmetric_lamarckian",  dict(mode="lamarckian", producer_rule="symmetric"),  list(range(30))),
    ("symmetric_baldwinian",  dict(mode="baldwinian", producer_rule="symmetric"),  list(range(30))),
]

if __name__ == "__main__":
    t0 = time.time()
    results = {}
    for name, kw, seeds in CONDITIONS:
        accs = [run_seed(sd, **kw) for sd in seeds]
        m, ci = ci95(accs)
        results[name] = {"mode": kw["mode"], "producer_rule": kw["producer_rule"],
                         "rounds_per_gen": 500, "n_seeds": len(seeds),
                         "final_acc_mean": m, "final_acc_ci95": ci, "final_accs": accs}
        print(f"{name:24s} n={len(seeds):2d}  final acc = {m:.3f} +/- {ci:.3f}"
              f"   [{time.time()-t0:.0f}s]", flush=True)
    with open("symmetric_rate_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDONE in {time.time()-t0:.0f}s -> symmetric_rate_results.json")
