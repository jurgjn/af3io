import json
from pathlib import PurePosixPath
import click.testing, numpy as np, pandas as pd, pytest, af3io, af3io.cli, af3io.predictions
from testdata import TEST_DATA, fetch_test_dir

# Every prediction in af3io-testdata: AlphaFold 3 examples (proteins, DNA, RNA, ligands, ions, modifications, glycans)
# and pooled predictions (~5k tokens, protein only)
PREDICTIONS = sorted( str(PurePosixPath(fname).parent) for fname in TEST_DATA.registry
    if af3io.predictions._is_ranking_scores(fname) )

# Chain pair scores copied from summary_confidences.json, which AlphaFold 3 can set to null (e.g. for ions)
NULLABLE = ['chain_pair_iptm', 'chain_pair_pae_min', 'chain_pair_iptm_corrected', 'chain_pair_model_confidence', 'chain_pair_model_confidence_corrected']

@pytest.mark.parametrize('prefix', PREDICTIONS)
def test_read_summary_scores(prefix):
    pred_dir = fetch_test_dir(prefix)
    df = af3io.predictions.read_summary_scores(pred_dir)
    assert len(df) == 5

    with af3io.predictions.Predictions(pred_dir) as pred:
        for _, row in df.iterrows():
            # Tokens from the model structure line up with the tokens in confidences.json
            with pred.open(row['confidences_path']) as fh:
                js = json.load(fh)
            tokens = af3io.predictions.get_model_tokens(af3io.predictions.read_struct(pred, row['model_path']))
            assert [ atoms[0].get_parent().id[1] for atoms in tokens ] == js['token_res_ids']

            n_chains = len(row['chain_ptm'])
            for col in df.columns:
                if col.startswith('chain_pair_'):
                    values = np.asarray(row[col], dtype=float)
                    assert values.shape == (n_chains, n_chains), col
                    if col not in NULLABLE:
                        assert np.isfinite(values).all(), col

def test_summary_confidences_all(tmp_path):
    for prefix in PREDICTIONS:
        fetch_test_dir(prefix)
    output_parquet = tmp_path / 'summary_confidences.parquet'
    result = click.testing.CliRunner().invoke(af3io.cli.summary_confidences, [str(TEST_DATA.abspath), str(output_parquet)])
    assert result.exit_code == 0, result.output
    assert f'{len(PREDICTIONS)} predictions found' in result.output
    df = pd.read_parquet(output_parquet)
    assert len(df) == 5 * len(PREDICTIONS)

    # Scores written to parquet match those computed in memory (a protein-only and a mixed protein/DNA example)
    for name in ['barnase_barstar', 'tetr_dimer_dna']:
        expected = af3io.predictions.read_summary_scores(fetch_test_dir(f'alphafold3_examples/alphafold3_predictions/{name}'))
        actual = df[df['name'] == name].reset_index(drop=True)
        for col in expected.columns:
            if col.startswith('chain_pair_'):
                for value_actual, value_expected in zip(actual[col], expected[col]):
                    np.testing.assert_array_equal(np.array([ list(row) for row in value_actual ], dtype=float), np.asarray(value_expected, dtype=float), err_msg=f'{name} {col}')
