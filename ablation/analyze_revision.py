import argparse
import csv
from hashlib import file_digest
import json
from math import sqrt
from pathlib import Path
from statistics import mean, stdev

from run_revision import CONDITIONS, FIELDS, N, N_CONTEXTS, SEEDS, TRIALS, final_panel, iso_timestamp

ARMS = tuple(arm for arm, *_ in CONDITIONS) + ('random_guess',)
PRIMARY_ARMS = ('genotype_only', 'random_guess')


def read_and_check(directory, seed):
    panel = final_panel(seed)
    counts = {arm: 0 for arm in ARMS}
    seen = set()
    previous_ns = 0
    rows = []
    path = directory / f'seed-{seed}.csv'
    with path.open(encoding='ascii', newline='') as file:
        reader = csv.DictReader(file)
        if tuple(reader.fieldnames) != FIELDS:
            raise ValueError(f'{path}: unexpected columns')
        for row in reader:
            trial = int(row['trial'])
            arm = row['arm']
            timestamp_ns = int(row['timestamp_unix_ns'])
            guess = int(row['decoded_context'])
            if not (0 <= guess < N_CONTEXTS and 1 <= trial <= TRIALS and arm in ARMS and int(row['seed']) == seed):
                raise ValueError(f'{path}: invalid trial: {row}')
            if timestamp_ns < previous_ns or row['timestamp'] != iso_timestamp(timestamp_ns):
                raise ValueError(f'{path}: invalid or unordered timestamp: {row}')
            index = arm, trial
            if index in seen:
                raise ValueError(f'{path}: duplicate {index}')
            producer, receiver, context, _ = panel[trial - 1]
            correct = int(guess == context)
            expected_pair = ('', '') if arm == 'random_guess' else (str(producer), str(receiver))
            if (int(row['context']) != context or int(row['correct']) != correct
                    or (row['producer'], row['receiver']) != expected_pair):
                raise ValueError(f'{path}: unmatched trial or invalid outcome: {row}')
            counts[arm] += correct
            rows.append(tuple(row[name] for name in FIELDS))
            previous_ns = timestamp_ns
            seen.add(index)
    if len(seen) != len(ARMS) * TRIALS:
        raise ValueError(f'{path}: incomplete seed')
    return counts, rows


def interval(values):
    center = mean(values)
    # 0.975 quantile of Student's t with 29 degrees of freedom.
    margin = 2.045229642132703 * stdev(values) / sqrt(len(values))
    return dict(estimate=center, ci_low=center - margin, ci_high=center + margin)


def write_telemetry(path, rows):
    rows.sort(key=lambda row: row[0])
    with path.open('w', encoding='ascii', newline='') as file:
        writer = csv.writer(file, lineterminator='\n')
        writer.writerow(FIELDS)
        writer.writerows(rows)
    with path.open('rb') as file:
        return {'rows': len(rows), 'sha256': file_digest(file, 'sha256').hexdigest()}


def main():
    parser = argparse.ArgumentParser(description='Validate telemetry and calculate SUB-0001 revision statistics.')
    parser.add_argument('trials_directory', type=Path)
    parser.add_argument('output_directory', type=Path)
    args = parser.parse_args()
    destination = args.output_directory
    destination.mkdir(parents=True)
    primary, secondary = [], []
    counts_by_seed = {}
    arm_index = FIELDS.index('arm')
    for seed in SEEDS:
        counts, rows = read_and_check(args.trials_directory, seed)
        counts_by_seed[seed] = counts
        for row in rows:
            (primary if row[arm_index] in PRIMARY_ARMS else secondary).append(row)
    primary_stats = write_telemetry(destination / 'coadaptive-fresh-primary.csv', primary)
    secondary_stats = write_telemetry(destination / 'coadaptive-fresh-controls.csv', secondary)
    accuracy = {arm: [100 * counts_by_seed[seed][arm] / TRIALS for seed in SEEDS] for arm in ARMS}
    differences = {
        arm: [100 * (counts_by_seed[seed]['genotype_only'] - counts_by_seed[seed][arm]) / TRIALS for seed in SEEDS]
        for arm in ARMS if arm != 'genotype_only'
    }
    with (destination / 'coadaptive-fresh-per-seed.csv').open('w', encoding='ascii', newline='') as file:
        writer = csv.writer(file, lineterminator='\n')
        writer.writerow(('seed', 'trials_per_arm', *(f'{arm}_correct' for arm in ARMS)))
        writer.writerows((seed, TRIALS, *(counts_by_seed[seed][arm] for arm in ARMS)) for seed in SEEDS)
    primary_threshold = 50
    summary = {
        'seeds': list(SEEDS),
        'trials_per_seed_arm': TRIALS,
        'accuracy_percent': {arm: interval(values) for arm, values in accuracy.items()},
        'genotype_only_minus_comparator_points': {arm: interval(values) for arm, values in differences.items()},
        'primary_threshold_points': primary_threshold,
        'primary_bar_cleared': interval(differences['random_guess'])['ci_low'] > primary_threshold,
        'primary_telemetry': primary_stats,
        'secondary_telemetry': secondary_stats,
    }
    (destination / 'coadaptive-fresh-summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
