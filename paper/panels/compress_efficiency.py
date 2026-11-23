"""
    Sizes of AlphaFold 3 predictions uncompressed, zipped, and in af3io containers with cumulatively more aggressive
    compression settings (see design/specs/test_compress_efficency_long.md), from data/test_compress_efficiency.parquet

        python panels/compress_efficiency.py  # writes panels/compress_efficiency.pdf and .png
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.ticker, pandas as pd, pyarrow.parquet as pq

PAPER = Path(__file__).parents[1]
DATA = PAPER / 'data' / 'test_compress_efficiency.parquet'

# Categories left to right
CATEGORIES = {
    'other': 'all remaining data',
    'model': 'models',
    'contact_probs': 'contact_probs',
    'pae': 'pae',
    'data': 'data pipeline (*_data.json)',
}
DATASETS = { 'alphafold3_examples': 'AlphaFold 3 examples', 'mgen_pools_5k_top3': 'pooled predictions (~5k tokens)' }

def format_mb(size):
    return f'{size:,.2f} MB' if size < 1 else f'{size:,.1f} MB' if size < 100 else f'{size:,.0f} MB'

def main():
    df = pd.read_parquet(DATA)
    attrs = json.loads(pq.read_schema(DATA).metadata.get(b'PANDAS_ATTRS', b'{}'))
    df['mb'] = df['bytes'] / df['n_predictions'] / 1e6
    datasets = [ dataset for dataset in DATASETS if dataset in set(df['dataset']) ]
    stages = list(dict.fromkeys(df['stage']))

    fig, axes = plt.subplots(len(datasets), 2, figsize=(10, 1 + 0.35 * len(stages) * len(datasets)), squeeze=False,
        width_ratios=[4, 1], layout='constrained')
    for row, dataset in zip(axes, datasets):
        table = df[df['dataset'] == dataset].pivot_table(index='stage', columns='category', values='mb', sort=False).reindex(index=stages, columns=list(CATEGORIES)).fillna(0)
        totals = table.sum(axis=1)
        ax_left, ax_right = row
        for ax in row:
            left = pd.Series(0.0, index=table.index)
            for category, label in CATEGORIES.items():
                ax.barh(stages, table[category], left=left, label=label)
                left += table[category]
            ax.invert_yaxis()
            ax.xaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:,g}')) # thousands separator
        # Broken x axis: compressed sizes on the left, the end of the uncompressed bar on the right
        ax_left.set_xlim(0, totals.drop('uncompressed').max() * 1.5)
        ax_right.set_xlim(totals['uncompressed'] * 0.7, totals['uncompressed'] * 1.25)
        ax_right.set_yticks([])
        ax_left.set_title(f'{DATASETS[dataset]}, {df.loc[df["dataset"] == dataset, "n_predictions"].iloc[0]} predictions', loc='left')

        # Totals at the end of each bar, and how many times smaller than zip
        for stage in stages:
            label = format_mb(totals[stage])
            if stage not in ('uncompressed', 'zip') and totals[stage] < totals['zip']:
                label += f' ({totals["zip"] / totals[stage]:.1f}x smaller than zip)'
            ax = ax_right if stage == 'uncompressed' else ax_left
            ax.annotate(label, (totals[stage], stage), xytext=(3, 0), textcoords='offset points', va='center')

    axes[-1][0].set_xlabel('size per prediction (MB)')
    fig.legend(*axes[0][0].get_legend_handles_labels(), loc='outside upper center', ncol=len(CATEGORIES))
    if attrs:
        fig.text(1, 0, f'{attrs.get("git_commit", "")[:12]} {attrs.get("date", "")}', ha='right', va='bottom')
    for suffix in ('pdf', 'png'):
        fig.savefig(Path(__file__).with_suffix(f'.{suffix}'))

if __name__ == '__main__':
    main()
