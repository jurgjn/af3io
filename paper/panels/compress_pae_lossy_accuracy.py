"""
    Scores of predictions compressed with lossy (x) and lossless (y) PAE matrices (see
    design/specs/test_compress_pae_lossy_accuracy_long.md), from data/test_compress_pae_lossy_accuracy.parquet

        python panels/compress_pae_lossy_accuracy.py  # writes panels/compress_pae_lossy_accuracy.pdf and .png
"""
import json, math, textwrap
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, numpy as np, pandas as pd, pyarrow.parquet as pq

PAPER = Path(__file__).parents[1]
DATA = PAPER / 'data' / 'test_compress_pae_lossy_accuracy.parquet'
DATASETS = { 'alphafold3_examples': 'AlphaFold 3 examples', 'mgen_pools_5k_top3': 'pooled predictions (~5k tokens)' }

def main():
    df = pd.read_parquet(DATA)
    attrs = json.loads(pq.read_schema(DATA).metadata.get(b'PANDAS_ATTRS', b'{}'))
    df = df[np.isfinite(df['lossless']) & np.isfinite(df['lossy'])]
    df = df[df['chain1'] != df['chain2']] # interface scores (between chains)
    same = df.groupby('score').apply(lambda df_: (df_['lossless'] == df_['lossy']).all(), include_groups=False)
    scores = [ score for score in sorted(set(df['score'])) if not same[score] ]
    n_cols = 4
    n_rows = math.ceil(len(scores) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows + 1), squeeze=False, layout='constrained')
    for ax, score in zip(axes.flat, scores):
        df_score = df[df['score'] == score]
        for dataset, label in DATASETS.items():
            df_ = df_score[df_score['dataset'] == dataset]
            ax.scatter(df_['lossy'], df_['lossless'], label=label)
        ax.axline((0, 0), slope=1, color='k', linestyle=':')
        error = (df_score['lossy'] - df_score['lossless']).abs().max()
        r = np.corrcoef(df_score['lossy'], df_score['lossless'])[0, 1]
        ax.set_title(f'{score}\nmax |error| {error:.4f}\nr = {r:.5f}')
        ax.set_xlabel('lossy')
        ax.set_ylabel('lossless')
    for ax in axes.flat[len(scores):]:
        ax.set_visible(False)
    fig.legend(*axes.flat[0].get_legend_handles_labels(), loc='outside upper center', ncol=len(DATASETS))
    exact = ', '.join(score for score in sorted(same.index) if same[score])
    stamp = f'\n{attrs.get("git_commit", "")[:12]} {attrs.get("date", "")}' if attrs else ''
    fig.supxlabel(textwrap.fill(f'Same with lossy and lossless PAE: {exact}', 140) + stamp)
    for suffix in ('pdf', 'png'):
        fig.savefig(Path(__file__).with_suffix(f'.{suffix}'))

if __name__ == '__main__':
    main()
