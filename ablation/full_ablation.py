#!/usr/bin/env python3
"""
full_ablation.py -- unified ablation harness producing ALL ablation numbers from a
SINGLE codebase with uniform treatment, so the asymmetric/Lamarckian baseline is one
consistent value across tables (fixing the cross-script 0.859-vs-0.881 discrepancy).

Adds, relative to baldwinian_ablation.py:
  - producer_rule in {asymmetric, symmetric}  (the §3.5 ablation)
  - gini coefficients (producer, receiver) recorded at the final generation, to test
    whether the sparsity asymmetry survives a SYMMETRIC learning rule (i.e. is it task
    geometry or an artifact of the asymmetric rule?)
  - fitness_mode in {sum, mean}  (robustness of selection to sum-vs-mean reward)

Model, learning rules, noise, mutation, selection are IDENTICAL to
baldwinian_ablation.py / experiment.py (eta=0.02, producer 0.25*eta success-only in the
asymmetric rule; mutation N(0,0.02); fitness-proportional resampling). Stdlib + numpy.
Additive: does not modify experiment.py, baldwinian_ablation.py, or any existing setup.
"""
import json, time
import numpy as np

N = 100; N_CONTEXTS = 10; SIGNAL_DIM = 4
ETA = 0.02; PRODUCER_ETA = 0.25 * ETA
MUTATION_STD = 0.02; INIT_SCALE = 0.10
GENERATIONS = 1000; NOISE_SIGMA = 0.10


def new_pop(rng):
    return [{"w": rng.normal(0.0, INIT_SCALE, size=(N_CONTEXTS, SIGNAL_DIM)),
             "geno": None, "fit": 0.0, "n": 0} for _ in range(N)]


def _seed_geno(pop):
    for a in pop: a["geno"] = a["w"].copy()


def gini(w):
    """Gini over absolute weights (flattened), 0=uniform .. 1=concentrated."""
    x = np.sort(np.abs(w).flatten())
    s = x.sum()
    if s == 0: return 0.0
    n = len(x)
    return float((2.0 * np.sum((np.arange(1, n + 1)) * x) / (n * s)) - (n + 1.0) / n)


def pop_gini(pop):
    return float(np.mean([gini(a["w"]) for a in pop]))


def estimate_accuracy(producers, receivers, rng, sigma, trials=2000):
    hit = 0
    for _ in range(trials):
        P = producers[rng.integers(0, N)]; R = receivers[rng.integers(0, N)]
        c = int(rng.integers(0, N_CONTEXTS)); s = P["w"][c]
        s_noisy = s + rng.normal(0.0, sigma, size=SIGNAL_DIM) if sigma > 0 else s
        hit += (int(np.argmax(R["w"] @ s_noisy)) == c)
    return hit / trials


def resample(pop, rng, mode, fitness_mode):
    eps = 1e-9
    if fitness_mode == "mean":
        raw = np.array([a["fit"] / a["n"] if a["n"] > 0 else 0.0 for a in pop], dtype=float)
    else:
        raw = np.array([a["fit"] for a in pop], dtype=float)
    fits = np.clip(raw, 0.0, None)
    probs = (fits + eps) / (fits.sum() + eps * len(pop))
    out = []
    for _ in range(len(pop)):
        parent = pop[int(rng.choice(len(pop), p=probs))]
        if mode == "lamarckian":
            w = parent["w"] + rng.normal(0.0, MUTATION_STD, size=parent["w"].shape)
            out.append({"w": w, "geno": w.copy(), "fit": 0.0, "n": 0})
        else:
            g = parent["geno"] + rng.normal(0.0, MUTATION_STD, size=parent["geno"].shape)
            out.append({"w": g.copy(), "geno": g, "fit": 0.0, "n": 0})
    return out


def run_seed(seed, mode, producer_rule, rounds_per_gen=500, fitness_mode="sum",
             generations=GENERATIONS, sigma=NOISE_SIGMA, final_trials=2000):
    rng = np.random.default_rng(seed)
    producers, receivers = new_pop(rng), new_pop(rng)
    _seed_geno(producers); _seed_geno(receivers)
    last_acc = 0.0; gp = gr = 0.0
    for gen in range(generations):
        if mode != "lamarckian":
            for a in producers: a["w"] = a["geno"].copy()
            for a in receivers: a["w"] = a["geno"].copy()
        for a in producers: a["fit"] = 0.0; a["n"] = 0
        for a in receivers: a["fit"] = 0.0; a["n"] = 0
        for _ in range(rounds_per_gen):
            P = producers[rng.integers(0, N)]; R = receivers[rng.integers(0, N)]
            c = int(rng.integers(0, N_CONTEXTS)); s = P["w"][c].copy()
            s_noisy = s + rng.normal(0.0, sigma, size=SIGNAL_DIM) if sigma > 0 else s
            guess = int(np.argmax(R["w"] @ s_noisy))
            reward = 1 if guess == c else 0
            # receiver: full eta, hit+miss (unchanged)
            if guess == c:
                R["w"][c] += ETA * (s_noisy - R["w"][c])
            else:
                R["w"][guess] -= ETA * (s_noisy - R["w"][guess])
                R["w"][c]     += ETA * (s_noisy - R["w"][c])
            # producer
            if producer_rule == "asymmetric":
                if reward == 1:
                    P["w"][c] += PRODUCER_ETA * (s_noisy - P["w"][c])
            else:  # symmetric: full eta, every interaction
                P["w"][c] += ETA * (s_noisy - P["w"][c])
            P["fit"] += reward; R["fit"] += reward
            P["n"] += 1; R["n"] += 1
        if gen == generations - 1:
            last_acc = estimate_accuracy(producers, receivers, rng, sigma, trials=final_trials)
            gp, gr = pop_gini(producers), pop_gini(receivers)
        producers = resample(producers, rng, mode, fitness_mode)
        receivers = resample(receivers, rng, mode, fitness_mode)
    return last_acc, gp, gr


def ci95(v):
    a = np.array(v, float); return float(a.mean()), float(1.96 * a.std(ddof=1) / np.sqrt(len(a)))


CONDITIONS = [
    ("asym_lam_500",  dict(mode="lamarckian", producer_rule="asymmetric", rounds_per_gen=500),  list(range(30))),
    ("asym_bald_500", dict(mode="baldwinian", producer_rule="asymmetric", rounds_per_gen=500),  list(range(30))),
    ("asym_bald_2500",dict(mode="baldwinian", producer_rule="asymmetric", rounds_per_gen=2500), list(range(15))),
    ("sym_lam_500",   dict(mode="lamarckian", producer_rule="symmetric",  rounds_per_gen=500),  list(range(30))),
    ("sym_bald_500",  dict(mode="baldwinian", producer_rule="symmetric",  rounds_per_gen=500),  list(range(30))),
    ("asym_lam_500_meanfit", dict(mode="lamarckian", producer_rule="asymmetric", rounds_per_gen=500, fitness_mode="mean"), list(range(30))),
]

if __name__ == "__main__":
    t0 = time.time(); results = {}
    for name, kw, seeds in CONDITIONS:
        accs, gps, grs = [], [], []
        for sd in seeds:
            a, gp, gr = run_seed(sd, **kw); accs.append(a); gps.append(gp); grs.append(gr)
        m, ci = ci95(accs)
        results[name] = {**{k: v for k, v in kw.items()}, "n_seeds": len(seeds),
                         "acc_mean": m, "acc_ci95": ci,
                         "gini_producer": float(np.mean(gps)), "gini_receiver": float(np.mean(grs)),
                         "accs": accs}
        print(f"{name:24s} acc={m:.3f}+/-{ci:.3f}  giniP={np.mean(gps):.3f} giniR={np.mean(grs):.3f}"
              f"  [{time.time()-t0:.0f}s]", flush=True)
    with open("full_ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDONE in {time.time()-t0:.0f}s -> full_ablation_results.json")
