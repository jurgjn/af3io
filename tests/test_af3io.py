
# pytest tests/ --capture=no --disable-warnings

import collections, functools, gzip, hashlib, importlib.util, io, json, os, runpy, sys, tarfile, zipfile
try:
    from compression import zstd # Python >= 3.14
except ImportError:
    from backports import zstd
from pathlib import Path, PurePosixPath
import click, click.testing, numpy as np, pandas as pd, pooch, pytest, af3io, af3io.cli
import af3io.scores

def md5sum(file):
    return hashlib.md5(open(file, 'rb').read()).hexdigest()

# Test data is downloaded once and cached by pooch (default: ~/.cache/af3io-test-data, override with $AF3IO_TEST_DATA)
TESTDATA_COMMIT = 'e950c090352e808262f3280a888dbe1271f53616' # https://github.com/jurgjn/af3io-testdata
TEST_DATA = pooch.create(
    path=pooch.os_cache('af3io-test-data'),
    base_url=f'https://raw.githubusercontent.com/jurgjn/af3io-testdata/{TESTDATA_COMMIT}/',
    env='AF3IO_TEST_DATA',
)
TEST_DATA.load_registry(Path(__file__).with_name('test_data_registry.txt'))

def fetch_test_data(fname, tmpdir):
    # Symlink cached file into tmpdir as tests write output files next to their inputs
    path = Path(tmpdir) / Path(fname).name
    path.symlink_to(TEST_DATA.fetch(fname))
    return path

def fetch_test_dir(prefix):
    # Fetch all files under prefix, return the (read-only) cached directory
    for fname in TEST_DATA.registry:
        if fname.startswith(prefix + '/'):
            TEST_DATA.fetch(fname)
    return Path(TEST_DATA.abspath) / prefix

@pytest.fixture(scope='session')
def example_predictions_dir():
    # Pooled prediction (~5k tokens, protein only) as an output directory with every file compressed with zstd
    return fetch_test_dir('mgen_pools_5k_top3/pools_5k_0040f80')

@pytest.fixture(scope='session')
def example_zip(example_predictions_dir, tmp_path_factory):
    # Same prediction as a plain zip archive (decompressed files)
    zip_path = tmp_path_factory.mktemp('af3io_data') / 'pools_5k_0040f80.zip'
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
        for path in sorted(example_predictions_dir.rglob('*.zst')):
            name = PurePosixPath('pools_5k_0040f80', path.relative_to(example_predictions_dir).as_posix()).with_suffix('')
            zf.writestr(str(name), zstd.decompress(path.read_bytes()))
    return zip_path

@pytest.fixture(scope='session')
def example_predictions_zip(example_zip):
    return str(example_zip)

@pytest.fixture(scope='session')
def example_confidences_json(example_predictions_dir, tmp_path_factory):
    path = tmp_path_factory.mktemp('af3io_confidences') / 'pools_5k_0040f80_confidences.json'
    path.write_bytes(zstd.decompress((example_predictions_dir / 'pools_5k_0040f80_confidences.json.zst').read_bytes()))
    return str(path)

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

def test_pipeline_sequence():
    # modified_rna example: residues at modified positions are replaced by the parent nucleotide (OMG=>G, PSU=>U, 5MC=>C)
    rna_mods = [{'modificationType': 'PSU', 'basePosition': 13}, {'modificationType': '5MC', 'basePosition': 18}, {'modificationType': 'OMG', 'basePosition': 4}]
    assert af3io.residue_names.rna_sequence('GGCCCGAUAGCUCAGUCGGUAGAGC', rna_mods) == 'GGCGCGAUAGCUUAGUCCGUAGAGC'
    assert af3io.residue_names.rna_sequence('GGCXT') == 'GGCNN' # unknown RNA residues
    # Protein: phosphothreonine/-tyrosine (TPO/PTR) on T/Y are unchanged, unknown residues (e.g. selenocysteine U) => X
    ptms = [{'ptmType': 'TPO', 'ptmPosition': 2}, {'ptmType': 'PTR', 'ptmPosition': 4}]
    assert af3io.residue_names.protein_sequence('ATGYU', ptms) == 'ATGYX'
    assert af3io.residue_names.protein_sequence('AAAA', ptms) == 'ATAY'
    # DNA is written as-is by the data pipeline
    dna = {'sequence': 'GATTACA', 'modifications': [{'modificationType': '5CM', 'basePosition': 1}]}
    assert af3io.residue_names.pipeline_sequence('dna', dna) == 'GATTACA'

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

def _write_named_json(path, name):
    """ Write a minimal input JSON to path with the name attribute set verbatim (bypassing inference in af3io.input.write) """
    js = af3io.input.init(name=name)
    js['sequences'].append(af3io.input.init_sequence('protein', 'A', 'MKTFFVAGL'))
    with open(path, 'w') as fh:
        fh.write(af3io.input.dumps(js))

def test_fixname(tmp_path):
    _write_named_json(tmp_path / 'mismatch.json', 'other')
    _write_named_json(tmp_path / 'foo_v2.json', 'foo') # name is a prefix of the file name
    _write_named_json(tmp_path / 'x_data.json', 'x_data') # _data suffix is stripped
    _write_named_json(tmp_path / 'matching.json', 'matching')
    _write_named_json(tmp_path / 'Bad Name.json', 'bad_name')
    (tmp_path / 'compressed.json.gz').write_bytes(b'')
    mtime_matching = os.path.getmtime(tmp_path / 'matching.json')

    runner = click.testing.CliRunner()
    result = runner.invoke(af3io.cli.fixname, [str(p) for p in sorted(tmp_path.iterdir())])
    assert result.exit_code == 0
    assert '5 checked, 3 updated, 1 unsanitised, 1 skipped' in result.output

    assert af3io.input.read(str(tmp_path / 'mismatch.json'))['name'] == 'mismatch'
    assert af3io.input.read(str(tmp_path / 'foo_v2.json'))['name'] == 'foo_v2'
    assert af3io.input.read(str(tmp_path / 'x_data.json'))['name'] == 'x'
    assert af3io.input.read(str(tmp_path / 'Bad Name.json'))['name'] == 'bad_name'
    assert os.path.getmtime(tmp_path / 'matching.json') == mtime_matching

def test_fixname_check(tmp_path):
    _write_named_json(tmp_path / 'mismatch.json', 'other')
    _write_named_json(tmp_path / 'matching.json', 'matching')
    md5_before = md5sum(tmp_path / 'mismatch.json')

    runner = click.testing.CliRunner()
    result = runner.invoke(af3io.cli.fixname, ['--check', str(tmp_path / 'mismatch.json'), str(tmp_path / 'matching.json')])
    assert result.exit_code == 1
    assert md5sum(tmp_path / 'mismatch.json') == md5_before

    result = runner.invoke(af3io.cli.fixname, ['--check', str(tmp_path / 'matching.json')])
    assert result.exit_code == 0

# Real AlphaFold3 example (AURKA + TPX2, PDB-derived) distributed with the ipSAE reference
# implementation; pinned to a commit (see test_data_registry.txt) so the fixture and its published reference numbers
# (https://github.com/DunbrackLab/IPSAE/blob/main/Example/fold_aurka_0_tpx2_0_model_0_10_10.txt)
# stay in sync: ipSAE=0.448952/0.866498 (A->B/B->A), pDockQ=0.5235, pDockQ2=0.7120/0.6278
@pytest.fixture(scope='session')
def ipsae_example_dir(tmp_path_factory):
    tmpdir = tmp_path_factory.mktemp('ipsae_example')
    for fname in ['ipsae/ipsae.py', 'ipsae/fold_aurka_0_tpx2_0_full_data_0.json', 'ipsae/fold_aurka_0_tpx2_0_model_0.cif']:
        fetch_test_data(fname, tmpdir)
    return tmpdir

@pytest.fixture(scope='session')
def ipsae_reference(ipsae_example_dir):
    """ Run the published ipsae.py on the example AF3 prediction to get its internal, per-residue
    arrays (chains/pae_matrix/distances/cb_plddt) and its own computed scores. AlphaFold3 tokenises
    modified residues (e.g. the two phosphothreonines here) and ligands atom-by-atom; ipsae.py
    collapses these down to one token per real polymer residue (dropping ligand chains), which
    af3io.scores's chain_pair_reduce does not do on its own -- so its arrays, rather than the raw
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
    ipsae_mat = af3io.scores.chain_pair_reduce(functools.partial(af3io.scores.ipsae, pae_cutoff=10), chains, pae_matrix)
    assert ipsae_mat[0, 1] == pytest.approx(0.448952, abs=1e-5)  # A -> B, asym
    assert ipsae_mat[1, 0] == pytest.approx(0.866498, abs=1e-5)  # B -> A, asym
    assert af3io.scores.ptm_symm(ipsae_mat)[0, 1] == pytest.approx(0.866498, abs=1e-3)  # max
    assert af3io.scores.min_symm(ipsae_mat)[0, 1] == pytest.approx(0.448952, abs=1e-3)  # min

def test_interface_scores_synthetic():
    # Two chains (A: 3 tokens, B: 2 tokens); the A/B interface is a confident contact (A3-B1, PAE 3),
    # a confident non-contact (A2-B1, PAE 6) and an unconfident contact (A3-B2, PAE 20)
    chains = np.array(['A', 'A', 'A', 'B', 'B'])
    pae = np.full((5, 5), 30.0)
    pae[:3, :3] = pae[3:, 3:] = 1.0
    pae[2, 3] = pae[3, 2] = 3.0
    pae[1, 3] = pae[3, 1] = 6.0
    pae[2, 4] = pae[4, 2] = 20.0
    contacts = np.zeros((5, 5), dtype=bool)
    contacts[2, 3] = contacts[3, 2] = contacts[2, 4] = contacts[4, 2] = True

    def ab(func, *arrs):
        return af3io.scores.chain_pair_reduce(func, chains, *arrs)[0, 1]

    # LIS transform: 1 - PAE/12 for PAE < 12; cLIS/cLIA only count confident contacts
    assert ab(af3io.scores.lis, pae) == pytest.approx(((1 - 3/12) + (1 - 6/12)) / 2)
    assert ab(af3io.scores.clis, contacts, pae) == pytest.approx(1 - 3/12)
    assert ab(af3io.scores.lia, pae) == 2
    assert ab(af3io.scores.clia, contacts, pae) == 1
    lia_symm = af3io.scores.sum_symm(af3io.scores.chain_pair_reduce(af3io.scores.lia, chains, pae))
    clia_symm = af3io.scores.sum_symm(af3io.scores.chain_pair_reduce(af3io.scores.clia, chains, contacts, pae))
    assert af3io.scores.ilia(lia_symm, clia_symm)[0, 1] == pytest.approx(np.sqrt(4 * 2))

    # As in AFM-LIS, PAE exactly at the cutoff is confident (counted by LIA) but contributes 0 to LIS
    pae_cutoff = pae.copy()
    pae_cutoff[0, 4] = 12.0
    assert ab(af3io.scores.lia, pae_cutoff) == 3
    assert ab(af3io.scores.lis, pae_cutoff) == pytest.approx(((1 - 3/12) + (1 - 6/12) + 0) / 3)

    # Interface size: 2 contacting pairs, involving A3 and B1/B2
    assert ab(af3io.scores.n_contacts, contacts) == 2
    assert ab(af3io.scores.n_interface_residues, contacts) == 3

    # Pair pTM over both chains (d0 from 5 tokens, clipped to 19), max over aligned tokens of the mean TM-transformed PAE
    d0 = 1.24 * (19 - 15) ** (1/3) - 1.8
    tm = 1 / (1 + (pae / d0) ** 2)
    chain_pair_ptm = af3io.scores.chain_pair_ptm_from_pae(chains, pae)
    assert chain_pair_ptm[0, 1] == chain_pair_ptm[1, 0] == pytest.approx(tm.mean(axis=1).max())
    assert chain_pair_ptm[0, 0] == pytest.approx(tm[:3, :3].mean(axis=1).max())
    assert af3io.scores.model_confidence(0.5, chain_pair_ptm)[0, 1] == pytest.approx(0.8 * 0.5 + 0.2 * chain_pair_ptm[0, 1])

    # Symmetrisation
    asym = np.array([[1.0, 0.2], [0.7, 1.0]])
    assert af3io.scores.min_symm(asym)[0, 1] == af3io.scores.min_symm(asym)[1, 0] == 0.2
    assert af3io.scores.ptm_symm(asym)[0, 1] == 0.7

@pytest.fixture(scope='session')
def afm_lis():
    # AFM-LIS reference implementation (lis.py) imported as a module
    spec = importlib.util.spec_from_file_location('afm_lis', TEST_DATA.fetch('afm-lis/lis.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def test_lis_afm_lis(afm_lis):
    # LIS/cLIS/iLIS/LIA/cLIA/iLIA should match the AFM-LIS reference implementation for every model of a heterodimer
    # prediction (barnase-barstar has PAE values of exactly 12.0, at the confident interface cutoff)
    pred_dir = fetch_test_dir('alphafold3_examples/alphafold3_predictions/barnase_barstar')
    df = af3io.predictions.read_summary_scores(pred_dir)
    assert len(df) == 5
    with af3io.predictions.Predictions(pred_dir) as pred:
        for _, row in df.iterrows():
            with pred.open(row['model_path']) as fh:
                struct_text = fh.read().decode()
            with pred.open(row['confidences_path']) as fh:
                pae = np.array(json.load(fh)['pae'])
            (expected,) = afm_lis.analyze_single_model(struct_text, pae, {}, 'cif', 'af3', row['confidences_path'], None)
            assert (expected['ci'], expected['cj']) == ('A', 'B')
            for col, key in [('lis', 'LIS'), ('clis', 'cLIS'), ('ilis', 'iLIS'), ('lia', 'LIA'), ('clia', 'cLIA'), ('ilia', 'iLIA')]:
                assert row[f'chain_pair_{col}'][0][1] == pytest.approx(expected[key], abs=1e-6), (row['model_path'], col)

def test_pdockq_pdockq2_literature(ipsae_reference):
    # https://github.com/DunbrackLab/IPSAE/blob/main/Example/fold_aurka_0_tpx2_0_model_0_10_10.txt
    chains, pae_matrix = ipsae_reference['chains'], ipsae_reference['pae_matrix']
    distances, cb_plddt = ipsae_reference['distances'], ipsae_reference['cb_plddt']
    isin_8A = distances <= 8.0
    plddt_row = np.broadcast_to(cb_plddt[:, None], pae_matrix.shape)
    plddt_col = np.broadcast_to(cb_plddt[None, :], pae_matrix.shape)

    pdockq_mat = af3io.scores.chain_pair_reduce(af3io.scores.pdockq, chains, isin_8A, plddt_row, plddt_col)
    assert pdockq_mat[0, 1] == pytest.approx(0.5235, abs=1e-4)  # direction-independent
    assert pdockq_mat[1, 0] == pytest.approx(0.5235, abs=1e-4)

    pdockq2_mat = af3io.scores.chain_pair_reduce(af3io.scores.pdockq2, chains, isin_8A, pae_matrix, plddt_row, plddt_col)
    assert pdockq2_mat[0, 1] == pytest.approx(0.7120, abs=1e-4)  # A -> B, asym
    assert pdockq2_mat[1, 0] == pytest.approx(0.6278, abs=1e-4)  # B -> A, asym
    assert af3io.scores.ptm_symm(pdockq2_mat)[0, 1] == pytest.approx(0.7120, abs=1e-3)  # max

def _zip_bytes(members, compression=zipfile.ZIP_DEFLATED):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', compression=compression) as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return buf.getvalue()

def _tar_bytes(members):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w') as tf:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()

def test_archive_relpath(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert af3io.archive.relpath(tmp_path / 'pools.tar') == 'pools.tar'
    assert af3io.archive.relpath(af3io.archive.join(tmp_path / 'sub' / 'pools.tar', 'dir/x.zip', 'x/y.cif')) == 'sub/pools.tar::dir/x.zip::x/y.cif'
    assert af3io.archive.relpath(af3io.archive.join(tmp_path.parent / 'pools.tar', 'x.zip')) == '../pools.tar::x.zip'
    assert af3io.archive.relpath('pools.tar::x.zip') == 'pools.tar::x.zip'

def test_archive_walk_open(tmp_path):
    # Leaf files (as stored, i.e. possibly compressed) and their decompressed contents
    leaves = { 'a/x.txt': b'x', 'a/y.json.gz': gzip.compress(b'y'), 'a/z.json.zst': zstd.compress(b'z') }
    decompressed = { 'a/x.txt': b'x', 'a/y.json.gz': b'y', 'a/z.json.zst': b'z' }
    inner_zip = _zip_bytes(leaves)
    zip_of_zips = _zip_bytes({ 'inner.zip': inner_zip, 'stored.zip': inner_zip }, compression=zipfile.ZIP_STORED)
    tar = _tar_bytes({
        'dir/inner.zip': inner_zip,
        'inner.zip.gz': gzip.compress(inner_zip),
        'inner.zip.zst': zstd.compress(inner_zip),
        'zips.zip': zip_of_zips,
    })
    for name, data in {
        'inner.zip': inner_zip,
        'zips.zip': zip_of_zips,
        'pools.tar': tar,
        'pools.tar.gz': gzip.compress(tar),
        'pools.tgz': gzip.compress(tar),
        'pools.tar.zst': zstd.compress(tar),
        'tar_in_zip.zip': _zip_bytes({ 'pools.tar': tar }),
    }.items():
        (tmp_path / name).write_bytes(data)

    def expected_(prefix, containers):
        return [ af3io.archive.join(prefix, *containers, leaf) for leaf in leaves ]

    tar_containers = [ ['dir/inner.zip'], ['inner.zip.gz'], ['inner.zip.zst'], ['zips.zip', 'inner.zip'], ['zips.zip', 'stored.zip'] ]
    expected = {
        'inner.zip': expected_(tmp_path / 'inner.zip', []),
        'zips.zip': expected_(tmp_path / 'zips.zip', ['inner.zip']) + expected_(tmp_path / 'zips.zip', ['stored.zip']),
        **{ name: [ loc for containers in tar_containers for loc in expected_(tmp_path / name, containers) ] for name in ['pools.tar', 'pools.tar.gz', 'pools.tgz', 'pools.tar.zst'] },
        'tar_in_zip.zip': [ loc for containers in tar_containers for loc in expected_(tmp_path / 'tar_in_zip.zip', ['pools.tar', *containers]) ],
    }
    for name, expected_locators in expected.items():
        locators = list(af3io.archive.walk(tmp_path / name))
        assert locators == expected_locators, name
        for locator in locators:
            with af3io.archive.open(locator) as fh:
                assert fh.read() == decompressed[af3io.archive.split(locator)[-1]], locator
            with af3io.archive.open(locator, seekable=True) as fh:
                fh.seek(0)
                assert fh.read() == decompressed[af3io.archive.split(locator)[-1]], locator

    # Directory traversal is sorted & recursive
    assert list(af3io.archive.walk(tmp_path)) == [ loc for name in sorted(expected) for loc in expected[name] ]

    with pytest.raises(KeyError):
        with af3io.archive.open(af3io.archive.join(tmp_path / 'pools.tar', 'missing.zip')):
            pass
    with pytest.raises(KeyError):
        with af3io.archive.open(af3io.archive.join(tmp_path / 'pools.tar.zst', 'missing.zip')):
            pass

@pytest.fixture(scope='session')
def example_predictions_tar(example_zip, tmp_path_factory):
    # Tar file with the example prediction twice: in a sub-directory, and compressed with zstd
    tar_path = tmp_path_factory.mktemp('af3io_tar') / 'pools.tar'
    data = example_zip.read_bytes()
    tar_path.write_bytes(_tar_bytes({ 'dir/pools_5k_0040f80.zip': data, 'pools_5k_0040f80.zip.zst': zstd.compress(data) }))
    return tar_path

@pytest.fixture(scope='session')
def example_predictions_dirs(example_zip, tmp_path_factory):
    # Unpacked example prediction with every file compressed individually, once with gzip, once with zstd
    root = tmp_path_factory.mktemp('af3io_dirs')
    with zipfile.ZipFile(example_zip) as zf:
        for suffix, compress in [('.gz', functools.partial(gzip.compress, compresslevel=1)), ('.zst', zstd.compress)]:
            for info in zf.infolist():
                if not info.is_dir():
                    path = root / suffix.lstrip('.') / (info.filename + suffix)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(compress(zf.read(info)))
    return [ root / 'gz' / 'pools_5k_0040f80', root / 'zst' / 'pools_5k_0040f80' ]

def test_predictions_dirs(example_predictions_zip, example_predictions_dirs):
    root = example_predictions_dirs[0].parent.parent
    assert list(af3io.predictions.find_predictions(root)) == [ str(path) for path in example_predictions_dirs ]
    assert list(af3io.predictions.find_predictions(example_predictions_dirs[0])) == [ str(example_predictions_dirs[0]) ]

    expected = af3io.predictions.read_summary_confidences(example_predictions_zip)
    for path in example_predictions_dirs:
        df = af3io.predictions.read_summary_confidences(path)
        assert (df['predictions_path'] == path).all()
        pd.testing.assert_frame_equal(df.drop(columns='predictions_path'), expected.drop(columns='predictions_path'))
        assert af3io.predictions.read_model(path) == af3io.predictions.read_model(example_predictions_zip)

def test_predictions_nested(example_predictions_zip, example_predictions_tar):
    locators = list(af3io.predictions.find_predictions(example_predictions_tar))
    assert locators == [
        af3io.archive.join(example_predictions_tar, 'dir/pools_5k_0040f80.zip'),
        af3io.archive.join(example_predictions_tar, 'pools_5k_0040f80.zip.zst'),
    ]
    assert list(af3io.predictions.find_predictions(example_predictions_zip)) == [ example_predictions_zip ]

    expected = af3io.predictions.read_summary_confidences(example_predictions_zip)
    for locator in locators:
        df = af3io.predictions.read_summary_confidences(locator)
        assert (df['predictions_path'] == locator).all()
        pd.testing.assert_frame_equal(df.drop(columns='predictions_path'), expected.drop(columns='predictions_path'))
        assert af3io.predictions.read_model(locator) == af3io.predictions.read_model(example_predictions_zip)

def test_summary_confidences(example_predictions_zip, example_predictions_tar, example_predictions_dirs, tmp_path):
    output_parquet = tmp_path / 'summary_confidences.parquet'
    runner = click.testing.CliRunner()
    result = runner.invoke(af3io.cli.summary_confidences, ['--jobs', '5', str(example_predictions_tar), example_predictions_zip, *map(str, example_predictions_dirs), str(output_parquet)])
    assert result.exit_code == 0, result.output
    assert '5 predictions found' in result.output

    df = pd.read_parquet(output_parquet)
    expected = af3io.predictions.read_summary_scores(Path(example_predictions_zip).absolute())
    assert list(df.columns) == list(expected.columns)
    assert list(df['predictions_path'].unique()) == [
        af3io.archive.join(os.path.relpath(example_predictions_tar), 'dir/pools_5k_0040f80.zip'),
        af3io.archive.join(os.path.relpath(example_predictions_tar), 'pools_5k_0040f80.zip.zst'),
        os.path.relpath(example_predictions_zip),
        *map(os.path.relpath, example_predictions_dirs),
    ]
    for _, df_ in df.groupby('predictions_path'):
        pd.testing.assert_frame_equal(df_.drop(columns='predictions_path').reset_index(drop=True), expected.drop(columns='predictions_path'), check_dtype=False)

def test_summary_confidences_stdout(example_predictions_zip):
    runner = click.testing.CliRunner()
    result = runner.invoke(af3io.cli.summary_confidences, ['--jobs', '1', example_predictions_zip, '-'])
    assert result.exit_code == 0, result.output
    assert '1 predictions found' in result.stderr
    assert 'predictions found' not in result.stdout

    df = pd.read_csv(io.StringIO(result.stdout), sep='\t')
    expected = af3io.predictions.read_summary_scores(Path(example_predictions_zip).absolute())
    assert list(df.columns) == list(expected.columns)
    assert len(df) == len(expected)
    pd.testing.assert_series_equal(df['ranking_score'], expected['ranking_score'])

def test_summary_confidences_empty(tmp_path):
    runner = click.testing.CliRunner()
    result = runner.invoke(af3io.cli.summary_confidences, [str(tmp_path), str(tmp_path / 'out.parquet')])
    assert result.exit_code != 0
    assert 'No predictions found' in result.output
