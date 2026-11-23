"""
    Sizes of AlphaFold 3 predictions uncompressed, zipped, and in af3io containers with cumulatively more aggressive
    compression settings (see design/specs/test_compress_efficency_long.md), from data/test_compress_efficiency.parquet

        python panels/compress_efficiency.py  # writes panels/compress_efficiency.pdf and .png
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, pandas as pd, pyarrow.parquet as pq, json

PAPER = Path(__file__).parents[1]
DATA = PAPER / 'data' / 'test_compress_efficiency.parquet'

# Categories left to right, categorical palette slots 1-5 (validated for adjacent pairs, see dataviz skill)
CATEGORIES = {
    'other': ('all remaining data', '#2a78d6'),
    'model': ('models', '#eb6834'),
    'contact_probs': ('contact_probs', '#1baf7a'),
    'pae': ('pae', '#eda100'),
    'data': ('data pipeline (*_data.json)', '#e87ba4'),
}
SURFACE, INK, INK_SECONDARY, MUTED, GRID, BASELINE = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
DATASETS = { 'alphafold3_examples': 'AlphaFold 3 examples', 'mgen_pools_5k_top3': 'pooled predictions (~5k tokens)' }

def format_mb(size):
    return f'{size:,.1f} MB' if size < 100 else f'{size:,.0f} MB'

def main():
    df = pd.read_parquet(DATA)
    attrs = json.loads(pq.read_schema(DATA).metadata[b'PANDAS_ATTRS']) if b'PANDAS_ATTRS' in pq.read_schema(DATA).metadata else {}
    df['mb'] = df['bytes'] / df['n_predictions'] / 1e6
    datasets = [ dataset for dataset in DATASETS if dataset in set(df['dataset']) ]
    stages = list(dict.fromkeys(df['stage']))

    plt.rcParams.update({ 'font.family': 'sans-serif', 'font.size': 8, 'axes.edgecolor': BASELINE, 'axes.labelcolor': INK_SECONDARY,
        'xtick.color': MUTED, 'ytick.color': INK_SECONDARY, 'text.color': INK, 'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE })
    fig, axes = plt.subplots(len(datasets), 2, figsize=(7.2, 0.9 + 0.32 * len(stages) * len(datasets)), squeeze=False,
        gridspec_kw={ 'width_ratios': [4, 1], 'wspace': 0.04, 'hspace': 0.45 })

    for row, dataset in zip(axes, datasets):
        table = df[df['dataset'] == dataset].pivot_table(index='stage', columns='category', values='mb', sort=False).reindex(index=stages, columns=list(CATEGORIES)).fillna(0)
        totals = table.sum(axis=1)
        compressed_max = totals.drop('uncompressed').max()
        ax_left, ax_right = row
        for ax in row:
            left = pd.Series(0.0, index=table.index)
            for category, (label, color) in CATEGORIES.items():
                ax.barh(range(len(stages)), table[category], left=left, height=0.62, color=color, edgecolor=SURFACE, linewidth=0.8, label=label)
                left += table[category]
            ax.invert_yaxis()
            ax.grid(axis='x', color=GRID, linewidth=0.5)
            ax.set_axisbelow(True)
            for side in ('top', 'right'):
                ax.spines[side].set_visible(False)
        # Broken x axis: compressed sizes on the left, the end of the uncompressed bar on the right
        ax_left.set_xlim(0, compressed_max * 1.45)
        ax_right.set_xlim(totals['uncompressed'] * 0.72, totals['uncompressed'] * 1.22)
        ax_right.spines['left'].set_visible(False)
        ax_right.tick_params(axis='y', left=False, labelleft=False)
        ax_left.set_yticks(range(len(stages)), stages)
        ax_left.tick_params(axis='y', length=0)
        for ax, x in ((ax_left, 1), (ax_right, 0)): # break marks
            ax.plot([x, x], [-0.02, 0.02], transform=ax.transAxes, color=BASELINE, linewidth=0.8, clip_on=False)
        ax_left.set_title(f'{DATASETS[dataset]}, {df.loc[df["dataset"] == dataset, "n_predictions"].iloc[0]} predictions', loc='left', fontsize=8, color=INK)

        # Totals at the end of each bar, and how many times smaller than zip
        for i, stage in enumerate(stages):
            total = totals[stage]
            label = format_mb(total)
            if stage not in ('uncompressed', 'zip') and total < totals['zip']:
                label += f'  ({totals["zip"] / total:.1f}x smaller than zip)'
            ax = ax_right if stage == 'uncompressed' else ax_left
            ax.text(total, i, f' {label}', va='center', ha='left', color=INK_SECONDARY, fontsize=7, clip_on=False)

    axes[-1][0].set_xlabel('size per prediction (MB)')
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.01, 1.0), ncol=len(CATEGORIES), frameon=False, fontsize=7, handlelength=1, columnspacing=1.2)
    if attrs:
        fig.text(0.99, 0.003, f'{attrs.get("git_commit", "")[:12]} {attrs.get("date", "")}', ha='right', va='bottom', color=MUTED, fontsize=5)
    fig.subplots_adjust(top=1 - 0.45 / fig.get_figheight(), left=0.2, right=0.98, bottom=0.35 / fig.get_figheight() + 0.04)
    for suffix in ('pdf', 'png'):
        fig.savefig(Path(__file__).with_suffix(f'.{suffix}'), dpi=200)

if __name__ == '__main__':
    main()
