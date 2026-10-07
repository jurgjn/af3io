
# pytest --capture=no --disable-warnings

import collections, functools, gzip, hashlib, io, os, runpy, sys, tarfile, zipfile
try:
    from compression import zstd # Python >= 3.14
except ImportError:
    from backports import zstd
from pathlib import Path
import click, click.testing, numpy as np, pandas as pd, pytest, af3io, af3io.cli
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
def example_predictions_tar(downloaded_zip, tmp_path_factory):
    # Tar file with the example prediction twice: in a sub-directory, and compressed with zstd
    tar_path = tmp_path_factory.mktemp('af3io_tar') / 'pools.tar'
    data = downloaded_zip.read_bytes()
    tar_path.write_bytes(_tar_bytes({ 'dir/pools_5k_0040f80.zip': data, 'pools_5k_0040f80.zip.zst': zstd.compress(data) }))
    return tar_path

@pytest.fixture(scope='session')
def example_predictions_dirs(downloaded_zip, tmp_path_factory):
    # Unpacked example prediction with every file compressed individually, once with gzip, once with zstd
    root = tmp_path_factory.mktemp('af3io_dirs')
    with zipfile.ZipFile(downloaded_zip) as zf:
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
