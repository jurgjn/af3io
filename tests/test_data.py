import collections, gzip
from pathlib import PurePosixPath
import click, click.testing, pytest, af3io, af3io.cli
from testdata import TEST_DATA, fetch_test_data

# AlphaFold 3 example inputs covering proteins, DNA, RNA, ligands (CCD codes, SMILES), ions, modifications, glycans
ALPHAFOLD3_EXAMPLES = sorted(PurePosixPath(fname).stem for fname in TEST_DATA.registry if fname.startswith('alphafold3_examples/alphafold3_jsons/'))

@pytest.fixture(scope='session')
def alphafold3_examples_data_dir(tmp_path_factory):
    # Data pipeline output (_data.json.gz) of all examples with a pre-computed sequence index
    data_dir = tmp_path_factory.mktemp('alphafold3_examples_data')
    for name in ALPHAFOLD3_EXAMPLES:
        fetch_test_data(f'alphafold3_examples/alphafold3_predictions/{name}/{name}_data.json.gz', data_dir)
    result = click.testing.CliRunner().invoke(af3io.cli.data_fill, ['--data_dir', str(data_dir), '--write-index'])
    assert result.exit_code == 0, result.output
    assert (data_dir / '.af3io_data_index.json').is_file()
    return data_dir

@pytest.mark.parametrize('name', ALPHAFOLD3_EXAMPLES)
def test_data_fill(name, alphafold3_examples_data_dir, tmp_path):
    """ Fill an AlphaFold 3 example input JSON with data pipeline output (looked up by sequence across all
    examples), the result should be identical to the actual data pipeline output.
    """
    json_path = fetch_test_data(f'alphafold3_examples/alphafold3_jsons/{name}.json', tmp_path)
    result = click.testing.CliRunner().invoke(af3io.cli.data_fill, [
        '--data_dir', str(alphafold3_examples_data_dir),
        '--json_path', str(json_path),
        '--output_dir', str(tmp_path),
    ])
    assert result.exit_code == 0, result.output
    data_json_expected = gzip.decompress((alphafold3_examples_data_dir / f'{name}_data.json.gz').read_bytes())
    assert (tmp_path / f'{name}_data.json').read_bytes() == data_json_expected

def test_data_fill_dna_chain(tmp_path):
    """ DNA chains have no MSA and no templates in data pipeline output; filling
    an input containing one must not fail, and must leave the protein path alone.
    """
    protein_seq = 'MKTFFVAGL'
    dna_seq = 'GTACTAGCAT'

    # Fake data pipeline output: protein carries an MSA and templates, DNA carries
    # neither, which is what AlphaFold 3 emits for nucleic acids.
    data_dir = tmp_path / 'data'
    data_dir.mkdir()

    js_protein = af3io.input.init(name='protein_chain')
    js_protein['sequences'].append(af3io.input.init_sequence('protein', 'A', protein_seq))
    js_protein['sequences'][0]['protein'].update({
        'unpairedMsa': f'>query\n{protein_seq}\n',
        'pairedMsa': '',
        'templates': [],
    })
    af3io.input.write(js_protein, str(data_dir / 'protein_chain_data.json'))

    js_dna = af3io.input.init(name='dna_chain')
    js_dna['sequences'].append(af3io.input.init_sequence('dna', 'B', dna_seq))
    af3io.input.write(js_dna, str(data_dir / 'dna_chain_data.json'))

    index = af3io.data.create_index(str(data_dir))

    # Protein + two DNA strands + a ligand
    js = af3io.input.init(name='complex')
    js['sequences'].append(af3io.input.init_sequence('protein', 'A', protein_seq))
    js['sequences'].append(af3io.input.init_sequence('dna', 'B', dna_seq))
    js['sequences'].append(af3io.input.init_sequence('dna', 'C', dna_seq))
    js['sequences'].append(collections.OrderedDict([('ligand', {'id': 'D', 'ccdCodes': ['ATP']})]))

    js_filled = af3io.data.fill(af3io.data.lookup(js, index))

    filled = {seq_fields['id']: seq_fields for _, seq_fields in af3io.input.iter_sequences(js_filled)}
    for field in ('unpairedMsa', 'pairedMsa', 'templates'):
        assert field in filled['A']
        assert field not in filled['B']
        assert field not in filled['C']
    assert filled['D']['ccdCodes'] == ['ATP']

def test_data_fill_protein_only_unchanged(tmp_path):
    """ Pin the protein path: all three fields are copied as before. """
    protein_seq = 'MKTFFVAGL'
    unpaired = f'>query\n{protein_seq}\n'

    data_dir = tmp_path / 'data'
    data_dir.mkdir()
    js_protein = af3io.input.init(name='protein_chain')
    js_protein['sequences'].append(af3io.input.init_sequence('protein', 'A', protein_seq))
    js_protein['sequences'][0]['protein'].update({
        'unpairedMsa': unpaired,
        'pairedMsa': '',
        'templates': [],
    })
    af3io.input.write(js_protein, str(data_dir / 'protein_chain_data.json'))

    index = af3io.data.create_index(str(data_dir))
    js = af3io.input.init(name='monomer')
    js['sequences'].append(af3io.input.init_sequence('protein', 'A', protein_seq))
    js_filled = af3io.data.fill(af3io.data.lookup(js, index))

    _, seq_fields = next(iter(af3io.input.iter_sequences(js_filled)))
    assert seq_fields['unpairedMsa'] == unpaired
    assert seq_fields['pairedMsa'] == ''
    assert seq_fields['templates'] == []
    assert 'dataPath' not in seq_fields
