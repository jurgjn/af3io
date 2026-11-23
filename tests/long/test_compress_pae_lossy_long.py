"""
    Lossy compression of PAE matrices (--compress-pae-lossy), see design/specs/test_compress_pae_lossy_accuracy_long.md,
    design/specs/test_compress_pae_lossy_bins_long.md, and design/adr/0006-compress-pae-lossy.md
"""
import hashlib
import click.testing, numpy as np, pandas as pd, pytest, af3io.cli, af3io.compress
from af3io.compress import confidences, matrix
from testdata import decompressed_contents, fetch_test_dir
from test_compress_long import DATASETS, confidences_parts

# Chain pair scores that only use PAE values up to 15 Å (or no PAE values), which are the same with lossy compression
EXACT = [
    'iptm', 'iptm_corrected', 'pae_min', # from summary_confidences.json
    'ipsae10', 'ipsae15', 'ipsae10_min', 'ipsae15_min', 'lis', 'lia', 'clis', 'clia', 'ilis', 'ilia',
    'pdockq', 'iplddt', 'contact_probs_max', 'contact_probs_pow3', 'contact_probs_pow6', 'contact_probs_pow9', 'n_contacts', 'n_interface_residues',
]

# Largest absolute difference between scores with lossless and lossy compression: about twice the largest difference in the
# test data, below the largest possible difference of a single pair of a pTM-like score (1/(1 + (PAE/d0)^2), 0.027 for
# d0 <= 12 Å), see design/adr/0006-compress-pae-lossy.md
MAX_ERROR = {
    'iptm_from_pae': 0.01,
    'actifptm_from_pae': 0.03,
    'actifpsae': 0.001,
    'reactifptm': 0.03,
    'model_confidence': 0.002,
    'model_confidence_corrected': 0.002,
    'pdockq2': 0.001,
    'pinc': 0.02,
}

def summary_confidences(image, output):
    result = click.testing.CliRunner().invoke(af3io.cli.summary_confidences, [str(image), str(output)])
    assert result.exit_code == 0, result.output
    return pd.read_parquet(output)

def chain_pair_scores(df):
    # One row per prediction, sample, score and pair of chains
    rows = []
    for column in df.columns:
        if column.startswith('chain_pair_'):
            for name, seed, sample, value in zip(df['name'], df['seed'], df['sample'], df[column]):
                values = np.array([ list(row) for row in value ], dtype=float)
                for (chain1, chain2), score in np.ndenumerate(values):
                    rows.append((name, seed, sample, column.removeprefix('chain_pair_'), chain1, chain2, score))
    return pd.DataFrame(rows, columns=['name', 'seed', 'sample', 'score', 'chain1', 'chain2', 'value'])

def test_compress_pae_lossy_accuracy(record, tmp_path):
    """Scores of predictions (summary-confidences) compressed with lossless and lossy PAE"""
    tables = []
    for dataset, prefix in DATASETS.items():
        source = fetch_test_dir(prefix)
        scores = {}
        for pae in ('lossless', 'lossy'):
            image = tmp_path / f'{dataset}_{pae}.af3io'
            af3io.compress.compress([source], image, af3io.compress.Settings(data='drop', pae=pae))
            scores[pae] = chain_pair_scores(summary_confidences(image, tmp_path / f'{dataset}_{pae}.parquet'))
        merged = scores['lossless'].merge(scores['lossy'], on=['name', 'seed', 'sample', 'score', 'chain1', 'chain2'], suffixes=('_lossless', '_lossy'), validate='1:1')
        assert len(merged) == len(scores['lossless']) == len(scores['lossy'])
        merged.insert(0, 'dataset', dataset)
        tables.append(merged.rename(columns={ 'value_lossless': 'lossless', 'value_lossy': 'lossy' }))

    df = record(pd.concat(tables))
    df['error'] = df['lossy'] - df['lossless']
    summary = df.groupby('score').agg(max_abs_error=('error', lambda error: error.abs().max()), mean_error=('error', 'mean'),
        r=('lossless', lambda lossless: np.corrcoef(lossless[np.isfinite(lossless)], df.loc[lossless.index, 'lossy'][np.isfinite(lossless)])[0, 1]))
    print(summary.to_string())
    assert set(df['score']) == set(EXACT) | set(MAX_ERROR)
    for score, df_score in df.groupby('score'):
        same = (df_score['lossless'] == df_score['lossy']) | (df_score['lossless'].isna() & df_score['lossy'].isna())
        if score in EXACT:
            assert same.all(), score
        else:
            assert df_score['error'].abs().max() <= MAX_ERROR[score], score
            assert summary.loc[score, 'r'] > 0.9999, score

def test_compress_pae_lossy_bins(record):
    """Distribution of PAE values in the test data, and their values after lossy compression"""
    rows = []
    for dataset, prefix in DATASETS.items():
        counts = np.zeros(1, dtype=np.int64)
        seen = set()
        for name, data in decompressed_contents(fetch_test_dir(prefix)).items():
            digest = hashlib.sha256(data).digest()
            if af3io.compress.classify(name) == 'confidences' and digest not in seen: # without copies of top-ranked models
                seen.add(digest)
                pae = confidences_parts(data)['pae']
                values = matrix.parse(pae[pae.index(b'['):].rstrip(b',\n '), 1)
                counts_ = np.bincount(values.ravel())
                counts = np.pad(counts, (0, max(0, len(counts_) - len(counts)))) + np.pad(counts_, (0, max(0, len(counts) - len(counts_))))
        values = np.arange(len(counts), dtype=np.uint16)
        lossy = confidences.quantize_pae(values)
        rows += [ { 'dataset': dataset, 'pae': value / 10, 'lossy': lossy_ / 10, 'count': count } for value, lossy_, count in zip(values.tolist(), lossy.tolist(), counts.tolist()) ]
    df = record(pd.DataFrame(rows))

    assert (df.loc[df['pae'] <= 15, 'pae'] == df.loc[df['pae'] <= 15, 'lossy']).all()
    assert set(df.loc[df['pae'] > 15, 'lossy']) == { value for _, _, value in confidences.PAE_LOSSY_BINS if value > 15 } & set(df.loc[df['pae'] > 15, 'lossy'])
    assert (df.loc[df['pae'] > 29, 'lossy'] == 31).all()
    fraction = df.assign(exact=df['pae'] <= 15).groupby(['dataset', 'exact'])['count'].sum() / df.groupby('dataset')['count'].sum()
    print(fraction)
    assert fraction['alphafold3_examples', True] > 0.8 # mostly small complexes with confident interfaces
    assert fraction['mgen_pools_5k_top3', False] > 0.9 # mostly pairs of proteins that do not interact
