#!/usr/bin/env python3
import os
import re
import sys
import glob
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt

# Simple analyzer for RLlib runs: finds progress.csv files and aggregates

def find_runs(root):
    paths = []
    for p in glob.glob(os.path.join(root, "**/progress.csv"), recursive=True):
        paths.append(p)
    return sorted(paths)


def load_progress(path):
    try:
        df = pd.read_csv(path)
        df['source'] = path
        return df
    except Exception as e:
        print(f"Failed to read {path}: {e}")
        return None


def extract_loss_columns(df):
    # Try new API first (learner_results.<module>.loss)
    loss_cols = [c for c in df.columns if c.endswith('.loss') and 'learner_results' in c]
    if not loss_cols:
        # Learner loss metric
        loss_cols = [c for c in df.columns if c.endswith('info/learner/loss')]
    return loss_cols


def summarize(df):
    loss_cols = extract_loss_columns(df)
    summary = {
        'iters': int(df['training_iteration'].max()) if 'training_iteration' in df else len(df),
        'env_steps_total': int(df.get('env_steps_total', pd.Series([0])).max()) if 'env_steps_total' in df else 0,
        'loss_col': loss_cols[0] if loss_cols else None,
        'final_loss': float(df[loss_cols[0]].dropna().iloc[-1]) if loss_cols and not df[loss_cols[0]].dropna().empty else None,
    }
    return summary, loss_cols


def plot_curves(dfs, outdir):
    os.makedirs(outdir, exist_ok=True)
    # Loss curves
    plt.figure(figsize=(10,6))
    found = False
    for df in dfs:
        loss_cols = extract_loss_columns(df)
        if not loss_cols:
            continue
        col = loss_cols[0]
        x = df['training_iteration'] if 'training_iteration' in df else range(len(df))
        plt.plot(x, df[col], alpha=0.8, label=os.path.basename(df['source'].iloc[0]))
        found = True
    if found:
        plt.xlabel('training_iteration')
        plt.ylabel('loss (MSE)')
        plt.title('Training Loss Curve(s)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, 'loss_curves.png'))
        plt.close()

    # Env steps curve
    plt.figure(figsize=(10,6))
    found = False
    for df in dfs:
        if 'env_steps_total' in df:
            x = df['training_iteration'] if 'training_iteration' in df else range(len(df))
            plt.plot(x, df['env_steps_total'], alpha=0.8, label=os.path.basename(df['source'].iloc[0]))
            found = True
    if found:
        plt.xlabel('training_iteration')
        plt.ylabel('env_steps_total')
        plt.title('Environment Steps Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, 'env_steps.png'))
        plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rllib-root', default=os.path.expanduser('~/ray_results'), help='Directory containing RLlib runs')
    ap.add_argument('--outdir', default='logs/processed', help='Where to save processed outputs')
    args = ap.parse_args()

    run_files = find_runs(args.rllib_root)
    if not run_files:
        print(f"No progress.csv found under {args.rllib_root}")
        sys.exit(0)

    dfs = []
    summaries = []
    for p in run_files:
        df = load_progress(p)
        if df is None:
            continue
        s, _ = summarize(df)
        s['source'] = p
        summaries.append(s)
        dfs.append(df)

    os.makedirs(args.outdir, exist_ok=True)
    # Save summary JSON
    with open(os.path.join(args.outdir, 'summary.json'), 'w') as f:
        json.dump(summaries, f, indent=2)
    # Concatenate CSV for quick filter
    pd.concat(dfs, ignore_index=True).to_csv(os.path.join(args.outdir, 'all_progress.csv'), index=False)

    # Plot curves
    plot_curves(dfs, args.outdir)

    # Print human summary
    print("Analyzed runs:\n")
    for s in summaries:
        print(json.dumps(s, indent=2))
    print(f"\nArtifacts saved to: {args.outdir}")

if __name__ == '__main__':
    main()
