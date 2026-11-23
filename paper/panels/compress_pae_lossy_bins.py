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
DATASETS = { 'alphafold3_examples': 'AlphaFold 3 examples', 'mgen_pools_5k_top3': 'pooled predictions (~5k tokens)' }
BINS = [ (15, 17, 16), (17, 19, 18), (19, 21, 20), (21, 23, 22), (23, 25, 24), (25, 27, 26), (27, 29, 28), (29, 32.5, 31) ] # see af3io.compress.confidences.PAE_LOSSY_BINS

def main():
    df = pd.read_parquet(DATA)
    attrs = json.loads(pq.read_schema(DATA).metadata.get(b'PANDAS_ATTRS', b'{}'))
    datasets = [ dataset for dataset in DATASETS if dataset in set(df['dataset']) ]
    fig, axes = plt.subplots(len(datasets) + 1, 1, figsize=(10, 2.5 * len(datasets) + 1.5), sharex=True,
        height_ratios=[3] * len(datasets) + [1], layout='constrained')

    for ax, dataset in zip(axes, datasets):
        df_ = df[df['dataset'] == dataset]
        fraction = df_['count'] / df_['count'].sum()
        ax.bar(df_['pae'], fraction, width=0.1)
        ax.set_yscale('log')
        ax.set_ylabel('fraction of PAE values')
        exact = fraction[df_['pae'] <= 15].sum()
        ax.set_title(f'{DATASETS[dataset]}: {100 * exact:.0f}% of values <= 15 Å (kept), {100 * (1 - exact):.0f}% in bins', loc='left')

    # Quantisation: values up to 15 Å are kept, larger values become the value of their bin
    ax = axes[-1]
    ax.add_patch(patches.Rectangle((0, 0), 15, 1, fill=False))
    ax.text(7.5, 0.5, 'lossless (values up to 15 Å kept)', ha='center', va='center')
    for lower, upper, value in BINS:
        ax.add_patch(patches.Rectangle((lower, 0), upper - lower, 1, fill=False))
        ax.axvline(value, ymax=1)
        ax.text(value, 1.05, f'{value}' + (' (final bin)' if value == BINS[-1][2] else ''), ha='center', va='bottom')
    ax.set_ylim(0, 1.5)
    ax.set_yticks([])
    ax.set_xlim(0, BINS[-1][1])
    ax.set_xlabel('PAE (Å)')
    if attrs:
        fig.text(1, 0, f'{attrs.get("git_commit", "")[:12]} {attrs.get("date", "")}', ha='right', va='bottom')
    for suffix in ('pdf', 'png'):
        fig.savefig(Path(__file__).with_suffix(f'.{suffix}'))

if __name__ == '__main__':
    main()
