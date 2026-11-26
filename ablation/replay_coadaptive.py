import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import gzip
import json
from math import sqrt
from pathlib import Path
from statistics import mean, stdev
from time import time_ns

import numpy as np

from baldwinian_ablation import run_seed

SEEDS = range(30)
FULL_FIELDS = ('timestamp_unix_ns', 'seed', 'stage', 'generation', 'trial', 'arm', 'context', 'decoded_context', 'correct')
FINAL_FIELDS = ('timestamp', *FULL_FIELDS)


def timestamp_iso(timestamp_ns):
    seconds, nanoseconds = divmod(timestamp_ns, 1_000_000_000)
    utc_seconds = datetime.fromtimestamp(seconds, timezone.utc).strftime('%Y-%m-%dT%H:%M:%S')
    return f'{utc_seconds}.{nanoseconds:09d}Z'


def run_one(seed, destination):
    # A separate RNG gives the baseline one real guess for every final context.
    baseline_rng = np.random.default_rng(np.random.SeedSequence([seed, 15820341]))
    full_name = destination / f'seed-{seed:02}.csv.gz'
    final_name = destination / f'final-{seed:02}.csv'
    baseline_hits = 0
    with gzip.open(full_name.with_suffix('.gz.part'), 'wt', encoding='ascii', newline='', compresslevel=1) as full_file, \
            final_name.with_suffix('.csv.part').open('w', encoding='ascii', newline='') as final_file:
        full = csv.writer(full_file, lineterminator='\n')
        final = csv.writer(final_file, lineterminator='\n')
        full.writerow(FULL_FIELDS)
        final.writerow(FINAL_FIELDS)

        def observe(stage, generation, trial, context, guess):
            nonlocal baseline_hits
            row = (time_ns(), seed, stage, generation, trial, 'genotype_only', context, guess, int(guess == context))
            full.writerow(row)
            if stage == 'final':
                final.writerow((timestamp_iso(row[0]), *row))
                random_guess = int(baseline_rng.integers(0, 10))
                correct = int(random_guess == context)
                baseline_hits += correct
                row = (time_ns(), seed, stage, generation, trial, 'random_guess', context, random_guess, correct)
                full.writerow(row)
                final.writerow((timestamp_iso(row[0]), *row))

        accuracy, _ = run_seed(seed, 'baldwinian', rounds_per_gen=500, observe=observe)
    final_name.with_suffix('.csv.part').rename(final_name)
    full_name.with_suffix('.gz.part').rename(full_name)
    return seed, accuracy, baseline_hits / 2000


def assemble_final_telemetry(destination):
    rows = []
    for seed in SEEDS:
        with (destination / f'final-{seed:02}.csv').open(encoding='ascii', newline='') as file:
            reader = csv.reader(file)
            next(reader)
            rows.extend(reader)
    rows.sort(key=lambda row: row[0])
    with (destination / 'coadaptive-final-telemetry.csv').open('w', encoding='ascii', newline='') as file:
        writer = csv.writer(file, lineterminator='\n')
        writer.writerow(FINAL_FIELDS)
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description='Replay the genotype-only signalling simulation for seeds 0–29.')
    parser.add_argument('destination', type=Path, help='New directory for trial logs and summary')
    parser.add_argument('workers', type=int, nargs='?', default=1, help='Number of worker processes (default: 1)')
    args = parser.parse_args()
    workers = args.workers
    if workers < 1:
        parser.error('workers must be at least 1')
    destination = args.destination.resolve()
    destination.mkdir(parents=True)
    seed_results = {}
    print(f'NumPy {np.__version__}, {workers} workers, seeds 0 to 29', flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_one, seed, destination) for seed in SEEDS]
        for future in as_completed(futures):
            seed, accuracy, baseline = future.result()
            seed_results[seed] = accuracy, baseline
            print(f'Seed {seed:02} complete: {accuracy:.4f} vs {baseline:.4f}', flush=True)
    assemble_final_telemetry(destination)

    model_percent = [100 * seed_results[seed][0] for seed in SEEDS]
    baseline_percent = [100 * seed_results[seed][1] for seed in SEEDS]
    gains = [model - baseline for model, baseline in zip(model_percent, baseline_percent)]
    estimate = mean(gains)
    # Two-sided 95% Student's t interval with 29 degrees of freedom.
    margin = 2.045229642132703 * stdev(gains) / sqrt(len(gains))
    summary = {
        'seed_count': len(SEEDS),
        'model_accuracy_percent': mean(model_percent),
        'baseline_accuracy_percent': mean(baseline_percent),
        'estimate': estimate,
        'ci_low': estimate - margin,
        'ci_high': estimate + margin,
        'unit': 'percentage points',
    }
    (destination / 'replay-results.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print('SPECTRO_RESULT ' + json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
