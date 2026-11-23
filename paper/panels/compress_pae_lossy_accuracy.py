"""
    Scores of predictions compressed with lossy (x) and lossless (y) PAE matrices (see
    design/specs/test_compress_pae_lossy_accuracy_long.md), from data/test_compress_pae_lossy_accuracy.parquet

        python panels/compress_pae_lossy_accuracy.py  # writes panels/compress_pae_lossy_accuracy.pdf and .png
"""
import json, math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, numpy as np, pandas as pd, pyarrow.parquet as pq

PAPER = Path(__file__).parents[1]
DATA = PAPER / 'data' / 'test_compress_pae_lossy_accuracy.parquet'
DATASETS = { 'alphafold3_examples': ('AlphaFold 3 examples', '#2a78d6'), 'mgen_pools_5k_top3': ('pooled predictions (~5k tokens)', '#eb6834') }
SURFACE, INK, INK_SECONDARY, MUTED, GRID, BASELINE = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'

def main():
    df = pd.read_parquet(DATA)
    attrs = json.loads(pq.read_schema(DATA).metadata.get(b'PANDAS_ATTRS', b'{}'))
    df = df[np.isfinite(df['lossless']) & np.isfinite(df['lossy'])]
    df = df[df['chain1'] != df['chain2']] # interface scores (between chains)
    same = df.groupby('score').apply(lambda df_: (df_['lossless'] == df_['lossy']).all(), include_groups=False)
    scores = [ score for score in sorted(set(df['score'])) if not same[score] ]
    plt.rcParams.update({ 'font.family': 'sans-serif', 'font.size': 7, 'axes.edgecolor': BASELINE, 'axes.labelcolor': INK_SECONDARY,
        'xtick.color': MUTED, 'ytick.color': MUTED, 'text.color': INK, 'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE })
    n_cols = 4
    n_rows = math.ceil(len(scores) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7.2, 1.85 * n_rows + 0.7), squeeze=False)
    for ax, score in zip(axes.flat, scores):
        df_score = df[df['score'] == score]
        for dataset, (label, color) in DATASETS.items():
            df_ = df_score[df_score['dataset'] == dataset]
            ax.scatter(df_['lossy'], df_['lossless'], s=6, color=color, edgecolors=SURFACE, linewidths=0.3, label=label, zorder=2)
        lims = [ min(df_score['lossy'].min(), df_score['lossless'].min()), max(df_score['lossy'].max(), df_score['lossless'].max()) ]
        ax.plot(lims, lims, color=BASELINE, linewidth=0.6, zorder=1)
        error = (df_score['lossy'] - df_score['lossless']).abs().max()
        r = np.corrcoef(df_score['lossy'], df_score['lossless'])[0, 1]
        ax.set_title(score, loc='left', fontsize=7, color=INK)
        ax.text(0.97, 0.05, f'max |error| {error:.4f}\nr = {r:.5f}', transform=ax.transAxes, ha='right', va='bottom', fontsize=6, color=INK_SECONDARY)
        ax.grid(color=GRID, linewidth=0.4)
        ax.set_axisbelow(True)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
    for ax in axes.flat[len(scores):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel('lossy')
    for ax in axes[:, 0]:
        ax.set_ylabel('lossless')
    handles, labels = axes.flat[0].get_legend_handles_labels()
    exact = ', '.join(score for score in sorted(same.index) if same[score])
    fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.01, 1.0), ncol=2, frameon=False, fontsize=7, markerscale=2)
    fig.text(0.01, 0.003, f'Same with lossy and lossless PAE: {exact}', ha='left', va='bottom', color=INK_SECONDARY, fontsize=5.5, wrap=True)
    if attrs:
        fig.text(0.99, 0.003, f'{attrs.get("git_commit", "")[:12]} {attrs.get("date", "")}', ha='right', va='bottom', color=MUTED, fontsize=5)
    fig.subplots_adjust(left=0.08, right=0.98, top=1 - 0.45 / fig.get_figheight(), bottom=0.15, hspace=0.55, wspace=0.35)
    for suffix in ('pdf', 'png'):
        fig.savefig(Path(__file__).with_suffix(f'.{suffix}'), dpi=200)

if __name__ == '__main__':
    main()
