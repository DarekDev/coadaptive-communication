import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
from pathlib import Path
from time import time_ns

import numpy as np

from baldwinian_ablation import N, N_CONTEXTS, NOISE_SIGMA, SIGNAL_DIM, run_seed

SEEDS = range(1000, 1030)
TRIALS = 2_000
FIELDS = ('timestamp', 'timestamp_unix_ns', 'seed', 'trial', 'arm', 'context', 'decoded_context', 'correct', 'producer', 'receiver')
CONDITIONS = (
    ('genotype_only', 'baldwinian', 1000, 500, True),
    ('learning_only', 'baldwinian', 1, 500, True),
    ('selection_only', 'baldwinian', 1000, 500, False),
    ('learned_weight_inheritance', 'lamarckian', 1000, 500, True),
    ('genotype_only_2500', 'baldwinian', 1000, 2500, True),
)


def iso_timestamp(ns):
    whole, fractional = divmod(ns, 1_000_000_000)
    return f"{datetime.fromtimestamp(whole, timezone.utc):%Y-%m-%dT%H:%M:%S}.{fractional:09d}Z"


def final_panel(seed):
    rng = np.random.default_rng(np.random.SeedSequence([seed, 104729]))
    sender = rng.integers(N, size=TRIALS)
    receiver = rng.integers(N, size=TRIALS)
    contexts = rng.integers(N_CONTEXTS, size=TRIALS)
    noise = rng.normal(0, NOISE_SIGMA, size=(TRIALS, SIGNAL_DIM))
    return tuple(zip(sender, receiver, contexts, noise, strict=True))


def run_one(seed, destination):
    panel = final_panel(seed)
    path = destination / f'seed-{seed}.csv'
    temporary = path.with_suffix('.csv.part')
    results = {}
    with temporary.open('w', encoding='ascii', newline='') as file:
        writer = csv.writer(file, lineterminator='\n')
        writer.writerow(FIELDS)

        def record(trial, arm, context, guess, producer='', receiver=''):
            ns = time_ns()
            correct = int(context == guess)
            writer.writerow((iso_timestamp(ns), ns, seed, trial, arm, context, guess, correct, producer, receiver))
            return correct

        for arm, mode, generations, rounds, learning in CONDITIONS:
            def evaluate(producers, receivers):
                hits = 0
                for trial, (p, r, context, noise) in enumerate(panel, 1):
                    signal = producers[p]['w'][context] + noise
                    guess = int(np.argmax(receivers[r]['w'] @ signal))
                    hits += record(trial, arm, context, guess, p, r)
                return hits / TRIALS

            accuracy, _ = run_seed(seed, mode, rounds, generations=generations,
                                   learning=learning, evaluate_final=evaluate)
            results[arm] = accuracy

        random_guess = np.random.default_rng(np.random.SeedSequence([seed, 15820341]))
        random_correct = sum(record(trial, 'random_guess', context, int(random_guess.integers(N_CONTEXTS)))
                             for trial, (_, _, context, _) in enumerate(panel, 1))
        results['random_guess'] = random_correct / TRIALS
    temporary.replace(path)
    return seed, results


def main():
    parser = argparse.ArgumentParser(description='Run the fresh-seed SUB-0001 revision experiment.')
    parser.add_argument('output_directory', type=Path, help='new directory for per-seed final-trial CSVs')
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be at least 1')
    destination = args.output_directory.resolve()
    destination.mkdir(parents=True)
    print(f'NumPy {np.__version__}; {args.workers} workers; seeds {SEEDS.start}–{SEEDS.stop - 1}', flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_one, seed, destination) for seed in SEEDS]
        for future in as_completed(futures):
            seed, results = future.result()
            print(f'Seed {seed} complete: ' + ', '.join(f'{arm}={accuracy:.4f}' for arm, accuracy in results.items()), flush=True)


if __name__ == '__main__':
    main()
