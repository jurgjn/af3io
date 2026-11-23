"""
    Distribution of PAE values in the test data, with the quantisation of lossy PAE compression (--compress-pae-lossy) below
    (see design/specs/test_compress_pae_lossy_bins_long.md), from data/test_compress_pae_lossy_bins.parquet

        python panels/compress_pae_lossy_bins.py  # writes panels/compress_pae_lossy_bins.pdf and .png
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.patches as patches, pandas as pd, pyarrow.parquet as pq

PAPER = Path(__file__).parents[1]
DATA = PAPER / 'data' / 'test_compress_pae_lossy_bins.parquet'
DATASETS = { 'alphafold3_examples': ('AlphaFold 3 examples', '#2a78d6'), 'mgen_pools_5k_top3': ('pooled predictions (~5k tokens)', '#eb6834') }
SURFACE, INK, INK_SECONDARY, MUTED, GRID, BASELINE = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
EXACT, BIN_LIGHT, BIN_DARK, FINAL = '#e1e0d9', '#cde2fb', '#9ec5f4', '#86b6ef'

def main():
    df = pd.read_parquet(DATA)
    attrs = json.loads(pq.read_schema(DATA).metadata.get(b'PANDAS_ATTRS', b'{}'))
    plt.rcParams.update({ 'font.family': 'sans-serif', 'font.size': 8, 'axes.edgecolor': BASELINE, 'axes.labelcolor': INK_SECONDARY,
        'xtick.color': MUTED, 'ytick.color': MUTED, 'text.color': INK, 'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE })
    datasets = [ dataset for dataset in DATASETS if dataset in set(df['dataset']) ]
    fig, axes = plt.subplots(len(datasets) + 1, 1, figsize=(7.2, 1.5 * len(datasets) + 1.1), sharex=True,
        gridspec_kw={ 'height_ratios': [3] * len(datasets) + [1.4], 'hspace': 0.25 })
    xmax = 32.5

    for ax, dataset in zip(axes, datasets):
        label, color = DATASETS[dataset]
        df_ = df[df['dataset'] == dataset]
        fraction = df_['count'] / df_['count'].sum()
        ax.bar(df_['pae'], fraction, width=0.1, align='center', color=color, linewidth=0)
        ax.set_yscale('log')
        ax.set_ylabel('fraction of\nPAE values')
        exact = fraction[df_['pae'] <= 15].sum()
        ax.set_title(f'{label}: {100 * exact:.0f}% of values <= 15 Å (kept), {100 * (1 - exact):.0f}% in bins', loc='left', fontsize=8, color=INK)
        for edge in [15, 17, 19, 21, 23, 25, 27, 29]:
            ax.axvline(edge, color=GRID, linewidth=0.6, zorder=0)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)

    # Quantisation track: kept values, 2 Å bins with the value they become, and the final bin
    ax = axes[-1]
    ax.set_ylim(0, 1)
    ax.add_patch(patches.Rectangle((0, 0.25), 15, 0.5, facecolor=EXACT, edgecolor=SURFACE, linewidth=1))
    ax.text(7.5, 0.5, 'lossless (all values up to 15 Å kept)', ha='center', va='center', color=INK_SECONDARY, fontsize=7)
    for i, (lower, upper, value) in enumerate([ (15, 17, 16), (17, 19, 18), (19, 21, 20), (21, 23, 22), (23, 25, 24), (25, 27, 26), (27, 29, 28) ]):
        ax.add_patch(patches.Rectangle((lower, 0.25), upper - lower, 0.5, facecolor=BIN_LIGHT if i % 2 == 0 else BIN_DARK, edgecolor=SURFACE, linewidth=1))
        ax.plot([value, value], [0.25, 0.75], color=INK_SECONDARY, linewidth=1)
        ax.text(value, 0.85, f'{value}', ha='center', va='bottom', color=INK_SECONDARY, fontsize=7)
    ax.add_patch(patches.Rectangle((29, 0.25), xmax - 29, 0.5, facecolor=FINAL, edgecolor=SURFACE, linewidth=1))
    ax.plot([31, 31], [0.25, 0.75], color=INK_SECONDARY, linewidth=1)
    ax.text(31, 0.85, '31', ha='center', va='bottom', color=INK_SECONDARY, fontsize=7)
    ax.text(30.75, 0.05, 'final bin', ha='center', va='bottom', color=INK_SECONDARY, fontsize=6.5)
    ax.text(22, 0.05, 'lossy: 2 Å bins, values become the bin centre', ha='center', va='bottom', color=INK_SECONDARY, fontsize=6.5)
    ax.set_yticks([])
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.set_xlim(0, xmax)
    ax.set_xlabel('PAE (Å)')
    if attrs:
        fig.text(0.99, 0.003, f'{attrs.get("git_commit", "")[:12]} {attrs.get("date", "")}', ha='right', va='bottom', color=MUTED, fontsize=5)
    fig.subplots_adjust(left=0.12, right=0.98, top=0.93, bottom=0.12)
    for suffix in ('pdf', 'png'):
        fig.savefig(Path(__file__).with_suffix(f'.{suffix}'), dpi=200)

if __name__ == '__main__':
    main()
