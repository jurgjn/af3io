
# pytest --capture=no --disable-warnings

import collections, functools, hashlib, os, runpy, sys
import click, click.testing, numpy as np, pytest, af3io, af3io.cli
import af3io.scoring

def md5sum(file):
    return hashlib.md5(open(file, 'rb').read()).hexdigest()

@pytest.fixture(scope='session')
def downloaded_zip(tmp_path_factory):
    tmpdir = tmp_path_factory.mktemp('af3io_data')
    zip_path = tmpdir / 'pools_5k_0040f80.zip'
    cmd = f'curl -s https://zenodo.org/records/16920556/files/pools_5k.tar?download=1 | tar -xC {tmpdir} -xf - --occurrence pools_5k_0040f80.zip'
    os.system(cmd)
    return zip_path

@pytest.fixture(scope='session')
def example_predictions_zip(downloaded_zip):
    return str(downloaded_zip)

@pytest.fixture(scope='session')
def example_confidences_json(downloaded_zip):
    zip_dir = downloaded_zip.parent
    os.system(f'unzip -qo {downloaded_zip} pools_5k_0040f80/pools_5k_0040f80_confidences.json -d {zip_dir}')
    fn = os.path.join(zip_dir, 'pools_5k_0040f80/pools_5k_0040f80_confidences.json')
    return fn

def test_confidences_compress_decompress(example_confidences_json):
    runner = click.testing.CliRunner()
    with runner.isolated_filesystem():
        # Copy the downloaded file to the isolated filesystem to avoid modifying the original or its directory
        import shutil
        local_json = 'example_confidences.json'
        shutil.copy(example_confidences_json, local_json)

        result_compress = runner.invoke(af3io.cli.confidences_compress, [local_json])
        assert result_compress.exit_code == 0

        result_decompress = runner.invoke(af3io.cli.confidences_decompress, [local_json + '.af3io'])
        assert result_decompress.exit_code == 0

        md5_downloaded = md5sum(local_json)
        md5_decompressed = md5sum(local_json + '.decompressed')
        assert md5_downloaded == md5_decompressed

@pytest.fixture(scope='session')
def example_data_json(downloaded_zip):
    zip_dir = downloaded_zip.parent
    os.system(f'unzip -qo {downloaded_zip} pools_5k_0040f80/pools_5k_0040f80_data.json -d {zip_dir}')
    fn = os.path.join(zip_dir, 'pools_5k_0040f80/pools_5k_0040f80_data.json')
    return fn

def test_data_fill(example_data_json):
    runner = click.testing.CliRunner()
    with runner.isolated_filesystem():
        data_dir = os.path.dirname(example_data_json)

        # We need the original sequences to create monomer jsons
        js = af3io.input.read(example_data_json)
        sequences = []
        for seq_type, seq_fields in af3io.input.iter_sequences(js):
            sequences.append(seq_fields['sequence'])

        # Create monomer input .json for every chain in the original pool
        os.makedirs('monomer_jsons', exist_ok=True)
        for i, seq in enumerate(sequences):
            chain_id = list(af3io.input.enumerate_chains())[i+1] # Starting from B as in notebook
            result = runner.invoke(af3io.cli.input_create, [f'monomer_jsons/chain_{chain_id.lower()}.json', '--sequence', seq])
            assert result.exit_code == 0

        # Copy data pipeline strings using af3io data-fill
        os.makedirs('monomer_msas', exist_ok=True)
        result = runner.invoke(af3io.cli.data_fill, [
            '--data_dir', data_dir,
            '--input_dir', 'monomer_jsons',
            '--output_dir', 'monomer_msas'
        ])
        assert result.exit_code == 0

        # Create "index" mapping available sequences to data pipeline .jsons
        result = runner.invoke(af3io.cli.data_fill, [
            '--data_dir', 'monomer_msas',
            '--write-index'
        ])
        assert result.exit_code == 0
        assert os.path.exists('monomer_msas/.af3io_data_index.json')

        # Create multimer input file with sequences from the original pool
        args = ['multimer_jsons/pools_5k_0040f80.json', '--model_seed', '4']
        for i, seq in enumerate(sequences):
            chain_id = list(af3io.input.enumerate_chains())[i+1]
            args.extend(['--type', 'protein', '--id', chain_id, '--sequence', seq])

        os.makedirs('multimer_jsons', exist_ok=True)
        result = runner.invoke(af3io.cli.input_create, args)
        assert result.exit_code == 0

        # Then fill in data pipeline strings from the monomers
        os.makedirs('multimer_msas', exist_ok=True)
        result = runner.invoke(af3io.cli.data_fill, [
            '--data_dir', 'monomer_msas',
            '--json_path', 'multimer_jsons/pools_5k_0040f80.json',
            '--output_dir', 'multimer_msas/'
        ])
        assert result.exit_code == 0

        # The result should be equal to what was downloaded from zenodo
        md5_original = md5sum(example_data_json)
        md5_filled = md5sum('multimer_msas/pools_5k_0040f80_data.json')
        assert md5_original == md5_filled

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

# Real AlphaFold3 example (AURKA + TPX2, PDB-derived) distributed with the ipSAE reference
# implementation; pinned to a commit so the fixture and its published reference numbers
# (https://github.com/DunbrackLab/IPSAE/blob/main/Example/fold_aurka_0_tpx2_0_model_0_10_10.txt)
# stay in sync: ipSAE=0.448952/0.866498 (A->B/B->A), pDockQ=0.5235, pDockQ2=0.7120/0.6278
IPSAE_COMMIT = '6174cf9e71cb1bd660cc805856a18c4871a6dec3'
IPSAE_RAW = f'https://raw.githubusercontent.com/DunbrackLab/IPSAE/{IPSAE_COMMIT}'

@pytest.fixture(scope='session')
def ipsae_example_dir(tmp_path_factory):
    tmpdir = tmp_path_factory.mktemp('ipsae_example')
    for fn in ['ipsae.py', 'Example/fold_aurka_0_tpx2_0_full_data_0.json', 'Example/fold_aurka_0_tpx2_0_model_0.cif']:
        dest = tmpdir / os.path.basename(fn)
        os.system(f'curl -s -o {dest} {IPSAE_RAW}/{fn}')
    return tmpdir

@pytest.fixture(scope='session')
def ipsae_reference(ipsae_example_dir):
    """ Run the published ipsae.py on the example AF3 prediction to get its internal, per-residue
    arrays (chains/pae_matrix/distances/cb_plddt) and its own computed scores. AlphaFold3 tokenises
    modified residues (e.g. the two phosphothreonines here) and ligands atom-by-atom; ipsae.py
    collapses these down to one token per real polymer residue (dropping ligand chains), which
    af3io.scoring's chain_pair_reduce does not do on its own -- so its arrays, rather than the raw
    full_data.json tokens, are the correct like-for-like input to compare af3io's scoring functions
    against the published per-chain-pair values.
    """
    json_path = str(ipsae_example_dir / 'fold_aurka_0_tpx2_0_full_data_0.json')
    cif_path = str(ipsae_example_dir / 'fold_aurka_0_tpx2_0_model_0.cif')
    saved_argv = sys.argv
    try:
        sys.argv = ['ipsae.py', json_path, cif_path, '10', '10']
        return runpy.run_path(str(ipsae_example_dir / 'ipsae.py'), run_name='ipsae_reference')
    finally:
        sys.argv = saved_argv

def test_ipsae_literature(ipsae_reference):
    # https://github.com/DunbrackLab/IPSAE/blob/main/Example/fold_aurka_0_tpx2_0_model_0_10_10.txt
    chains, pae_matrix = ipsae_reference['chains'], ipsae_reference['pae_matrix']
    ipsae_mat = af3io.scoring.chain_pair_reduce(functools.partial(af3io.scoring.ipsae, pae_cutoff=10), chains, pae_matrix)
    assert ipsae_mat[0, 1] == pytest.approx(0.448952, abs=1e-5)  # A -> B, asym
    assert ipsae_mat[1, 0] == pytest.approx(0.866498, abs=1e-5)  # B -> A, asym
    assert af3io.scoring.ptm_symm(ipsae_mat)[0, 1] == pytest.approx(0.866498, abs=1e-3)  # max

def test_lis_literature(ipsae_reference):
    # LIS isn't stable across ipsae.py versions (unlike ipSAE/pDockQ/pDockQ2 above), so compare
    # against ipsae.py's own LIS computation on this pinned commit rather than a hardcoded literature value
    chains, pae_matrix = ipsae_reference['chains'], ipsae_reference['pae_matrix']
    lis_mat = af3io.scoring.chain_pair_reduce(af3io.scoring.lis, chains, pae_matrix)
    assert lis_mat[0, 1] == pytest.approx(ipsae_reference['LIS']['A']['B'], abs=1e-6)
    assert lis_mat[1, 0] == pytest.approx(ipsae_reference['LIS']['B']['A'], abs=1e-6)

def test_pdockq_pdockq2_literature(ipsae_reference):
    # https://github.com/DunbrackLab/IPSAE/blob/main/Example/fold_aurka_0_tpx2_0_model_0_10_10.txt
    chains, pae_matrix = ipsae_reference['chains'], ipsae_reference['pae_matrix']
    distances, cb_plddt = ipsae_reference['distances'], ipsae_reference['cb_plddt']
    isin_8A = distances <= 8.0
    plddt_row = np.broadcast_to(cb_plddt[:, None], pae_matrix.shape)
    plddt_col = np.broadcast_to(cb_plddt[None, :], pae_matrix.shape)

    pdockq_mat = af3io.scoring.chain_pair_reduce(af3io.scoring.pdockq, chains, isin_8A, plddt_row, plddt_col)
    assert pdockq_mat[0, 1] == pytest.approx(0.5235, abs=1e-4)  # direction-independent
    assert pdockq_mat[1, 0] == pytest.approx(0.5235, abs=1e-4)

    pdockq2_mat = af3io.scoring.chain_pair_reduce(af3io.scoring.pdockq2, chains, isin_8A, pae_matrix, plddt_row, plddt_col)
    assert pdockq2_mat[0, 1] == pytest.approx(0.7120, abs=1e-4)  # A -> B, asym
    assert pdockq2_mat[1, 0] == pytest.approx(0.6278, abs=1e-4)  # B -> A, asym
    assert af3io.scoring.ptm_symm(pdockq2_mat)[0, 1] == pytest.approx(0.7120, abs=1e-3)  # max
