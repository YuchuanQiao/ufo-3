#!/usr/bin/env python3
"""Aggregate metrics.csv into a JSON summary and print a short report."""
import os
import csv
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--exp-dir', required=True)
parser.add_argument('--out', required=True)
args = parser.parse_args()

metrics_file = os.path.join(args.exp_dir, 'metrics.csv')
if not os.path.exists(metrics_file):
    print('No metrics.csv found in', args.exp_dir)
    exit(1)

rows = []
with open(metrics_file, 'r') as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append({k: float(v) if v not in ('', 'nan') else float('nan') for k, v in r.items()})

if not rows:
    print('No rows in metrics.csv')
    exit(1)

# Compute summary
best_val = max(rows, key=lambda x: x.get('val_acc', float('-inf')))
avg_epoch_time = sum([r.get('epoch_time', 0) for r in rows]) / len(rows)
summary = {
    'best_val_acc': best_val.get('val_acc', None),
    'best_val_epoch': best_val.get('epoch', None),
    'avg_epoch_time_seconds': avg_epoch_time,
    'num_epochs': len(rows)
}

with open(args.out, 'w') as f:
    json.dump({'summary': summary, 'rows': rows}, f, indent=2)

print('Wrote summary to', args.out)
print(summary)