import gzip, hashlib, io, itertools, json, os, shutil, subprocess, zipfile
try:
    from compression import zstd # Python >= 3.14
except ImportError:
    from backports import zstd
from pathlib import Path
import click.testing, numpy as np, pandas as pd, pytest, af3io, af3io.cli, af3io.compress, af3io.predictions
from af3io.compress import container, pipeline, squashfs
from testdata import decompressed_contents, fetch_test_dir, read_dir, tar_bytes, zip_bytes

@pytest.fixture(scope='session')
def alphafold3_examples_dir():
    # AlphaFold 3 example predictions with every file compressed with gzip
    return fetch_test_dir('alphafold3_examples/alphafold3_predictions')

@pytest.fixture(scope='session')
def alphafold3_examples_af3io(alphafold3_examples_dir, tmp_path_factory):
    path = tmp_path_factory.mktemp('af3io_compress') / 'alphafold3_examples.af3io'
    af3io.compress.compress([alphafold3_examples_dir], path, jobs=4)
    return path

def test_squashfs(tmp_path):
    # Image with an empty file, files spanning several blocks (exactly and with a tail end), and duplicates
    files = {
        'empty': b'',
        'a/small.txt': b'hello',
        'a/b/blocks.bin': np.random.default_rng(0).integers(0, 255, size=3 * (1 << 17) + 123, dtype=np.uint8).tobytes(),
        'a/b/exact.bin': bytes(range(256)) * (1 << 10),
        'a/copy.txt': b'hello',
        'z.txt': b'z' * 100_000,
    }
    image = tmp_path / 'test.af3io'
    with container.Writer(image, level=3, block_size=1 << 17) as writer:
        for name, data in files.items():
            writer.add(name, data, {'user.test.sha256': hashlib.sha256(data).hexdigest()})
        writer.add('noxattrs.txt', b'no xattrs')
    assert not any(path.name.startswith('.') for path in tmp_path.iterdir()) # temporary directory removed

    with squashfs.SquashFS(image) as fs:
        found = dict(fs.walk())
        assert list(found) == sorted(found, key=lambda path: path.split('/')) # depth-first, sorted by name
        assert set(found) == { f'/{name}' for name in files } | {'/noxattrs.txt'}
        for name, data in files.items():
            assert fs.read(found[f'/{name}']) == data
            assert fs.read(f'/{name}') == data
            assert fs.xattrs(found[f'/{name}']) == {'user.test.sha256': hashlib.sha256(data).hexdigest().encode()}
        assert fs.xattrs(found['/noxattrs.txt']) == {}
        assert fs.lookup('/a/b').isdir() and fs.lookup('/missing') is None and fs.lookup('/a/small.txt/x') is None
        assert list(dict(fs.walk('/a/b'))) == ['/a/b/blocks.bin', '/a/b/exact.bin']
        with pytest.raises(FileNotFoundError):
            fs.read('/missing')

        usage = fs.disk_usage(found.items())
        assert usage['/a/copy.txt'] == 0 or usage['/a/small.txt'] == 0 # duplicates are stored (and counted) once
        assert 0 < sum(usage.values()) < image.stat().st_size

    if shutil.which('unsquashfs') is not None: # matches squashfs-tools
        out = tmp_path / 'unsquashfs'
        subprocess.run(['unsquashfs', '-no-xattrs', '-d', str(out), str(image)], check=True, capture_output=True)
        assert read_dir(out) == { **files, 'noxattrs.txt': b'no xattrs' }

    with pytest.raises(FileExistsError):
        container.Writer(image)

def test_compress_inputs(tmp_path):
    # Archives and compressed files are replaced by their contents, sources are added under their base names
    root = tmp_path / 'root'
    (root / 'sub').mkdir(parents=True)
    (root / 'sub' / 'a.json.gz').write_bytes(gzip.compress(b'a'))
    (root / 'sub' / 'pred.zip').write_bytes(zip_bytes({ 'pred/pred_model.cif': b'model', 'pred/x/y.txt.zst': zstd.compress(b'y') }))
    (root / 'sub' / 'preds.tar').write_bytes(tar_bytes({ 'dir/p1.zip': zip_bytes({ 'p1/f.txt': b'f' }), 'g.txt': b'g' }))
    (tmp_path / 'single.txt.gz').write_bytes(gzip.compress(b'single'))
    sources = [root, tmp_path / 'single.txt.gz']
    assert [ path for path, _ in pipeline.iter_inputs(sources) ] == [
        'root/sub/a.json', 'root/sub/pred/pred_model.cif', 'root/sub/pred/x/y.txt',
        'root/sub/dir/p1/f.txt', 'root/sub/g.txt', 'single.txt',
    ]
    assert [ path for path, _ in pipeline.iter_inputs(sources, extract_archives=False) ] == [
        'root/sub/a.json.gz', 'root/sub/pred.zip', 'root/sub/preds.tar', 'single.txt.gz',
    ]

    # Duplicates after unpacking, e.g. a.json and a.json.gz
    (root / 'sub' / 'a.json').write_bytes(b'a')
    with pytest.raises(ValueError, match='duplicate file root/sub/a.json'):
        pipeline.group_inputs(pipeline.iter_inputs(sources))

    # Unsafe names in archives
    (tmp_path / 'unsafe.zip').write_bytes(zip_bytes({ '../evil.txt': b'evil' }))
    with pytest.raises(ValueError, match='unsafe'):
        list(pipeline.iter_inputs([tmp_path / 'unsafe.zip']))

def test_compress_roundtrip(alphafold3_examples_dir, alphafold3_examples_af3io, tmp_path):
    expected = decompressed_contents(alphafold3_examples_dir)
    names = af3io.compress.decompress(alphafold3_examples_af3io, tmp_path, jobs=2)
    assert names == sorted(expected)
    assert read_dir(tmp_path) == expected

    with af3io.compress.open(alphafold3_examples_af3io) as reader:
        assert reader.namelist() == sorted(expected)
        name = 'alphafold3_predictions/barnase_barstar/barnase_barstar_confidences.json'
        assert reader.read(name) == expected[name]
        assert reader.open(name).read() == expected[name]
        assert dict(reader.iter('alphafold3_predictions/barnase_barstar')) == { name: data for name, data in expected.items() if name.startswith('alphafold3_predictions/barnase_barstar/') }
        assert reader.manifest['af3io_version'] == af3io.__version__ if hasattr(af3io, '__version__') else True
        with pytest.raises(KeyError):
            reader.read('missing.json')

        # Sizes by category add up to the file size of the container
        df = reader.info()
        assert df.loc['total', 'compressed'] == pytest.approx(alphafold3_examples_af3io.stat().st_size, abs=10)
        assert df.loc['total', 'uncompressed'] == sum(map(len, expected.values()))
        assert df.loc['total', 'files'] == len(expected)
        assert df.loc['data', 'uncompressed'] == sum(len(data) for name, data in expected.items() if name.endswith('_data.json'))

def test_compress_reproducible(alphafold3_examples_dir, alphafold3_examples_af3io, tmp_path):
    path = tmp_path / 'again.af3io'
    af3io.compress.compress([alphafold3_examples_dir], path, jobs=3)
    assert path.read_bytes() == alphafold3_examples_af3io.read_bytes()

def test_compress_cli(alphafold3_examples_dir, tmp_path):
    runner = click.testing.CliRunner()
    source = alphafold3_examples_dir / 'barnase_barstar'
    image = tmp_path / 'barnase_barstar.af3io'
    files_from = tmp_path / 'files.txt'
    files_from.write_text(f'{source}\n')
    result = runner.invoke(af3io.cli.compress, ['--files-from', str(files_from), '-j', '2', str(image)])
    assert result.exit_code == 0, result.output
    expected = decompressed_contents(source)

    result = runner.invoke(af3io.cli.compress, [str(source), str(image)])
    assert result.exit_code != 0 and 'already exists' in result.output

    result = runner.invoke(af3io.cli.info, [str(image)])
    assert result.exit_code == 0, result.output
    assert 'confidences' in result.output and 'total' in result.output

    # Decompress selected files only
    result = runner.invoke(af3io.cli.decompress, [str(image), 'barnase_barstar/seed-42_sample-0/*', 'barnase_barstar/*.csv', str(tmp_path / 'out')])
    assert result.exit_code == 0, result.output
    assert read_dir(tmp_path / 'out') == { name: data for name, data in expected.items() if name.startswith('barnase_barstar/seed-42_sample-0/') or name.endswith('.csv') }

    # Existing files are not overwritten unless --force, the rest decompresses like a directory pattern
    result = runner.invoke(af3io.cli.decompress, [str(image), 'barnase_barstar', str(tmp_path / 'out')])
    assert result.exit_code != 0 and 'already exists' in result.output
    result = runner.invoke(af3io.cli.decompress, ['--force', '-j', '1', str(image), 'barnase_barstar', str(tmp_path / 'out')])
    assert result.exit_code == 0, result.output
    assert read_dir(tmp_path / 'out') == expected

    result = runner.invoke(af3io.cli.decompress, [str(image), 'missing/*', str(tmp_path / 'out')])
    assert result.exit_code != 0 and 'not found' in result.output

def test_compress_no_extract_archives(alphafold3_examples_dir, tmp_path):
    source = alphafold3_examples_dir / 'barnase_barstar'
    image = tmp_path / 'barnase_barstar.af3io'
    af3io.compress.compress([source], image, af3io.compress.Settings(extract_archives=False), jobs=2)
    af3io.compress.decompress(image, tmp_path / 'out', jobs=2)
    assert read_dir(tmp_path / 'out' / 'barnase_barstar') == read_dir(source)

def test_compress_checksum(tmp_path):
    # Files that do not match their checksum are not returned
    image = tmp_path / 'corrupt.af3io'
    manifest = { 'format': container.FORMAT_VERSION, 'af3io_version': '0', 'settings': {}, 'stats': { 'uncompressed': {} } }
    with container.Writer(image, level=3) as writer:
        writer.add(container.MANIFEST, json.dumps(manifest).encode())
        writer.add('contents/other/ok.txt', b'ok', { container.XATTR_SHA256: container.sha256(b'ok') })
        writer.add('contents/other/bad.txt', b'bad', { container.XATTR_SHA256: container.sha256(b'good') })
        writer.add('contents/other/none.txt', b'none')
    with af3io.compress.open(image) as reader:
        assert reader.read('ok.txt') == b'ok'
        with pytest.raises(af3io.compress.ChecksumError):
            reader.read('bad.txt')
        with pytest.raises(af3io.compress.ChecksumError):
            reader.read('none.txt')
        assert reader.read('bad.txt', verify=False) == b'bad'
    with pytest.raises(af3io.compress.ChecksumError):
        af3io.compress.decompress(image, tmp_path / 'out', ['bad.txt'], jobs=1)

def test_compress_predictions(alphafold3_examples_dir, alphafold3_examples_af3io):
    # Predictions read from a container match those read from the original directories
    locators = list(af3io.predictions.find_predictions(alphafold3_examples_af3io))
    assert locators == [ af3io.archive.join(alphafold3_examples_af3io, f'alphafold3_predictions/{path.name}') for path in sorted(alphafold3_examples_dir.iterdir()) ]
    for name in ['barnase_barstar', 'tetr_dimer_dna']:
        expected = af3io.predictions.read_summary_scores(alphafold3_examples_dir / name)
        actual = af3io.predictions.read_summary_scores(af3io.archive.join(alphafold3_examples_af3io, f'alphafold3_predictions/{name}'))
        pd.testing.assert_frame_equal(actual.drop(columns=['predictions_path']), expected.drop(columns=['predictions_path']))

    # Streaming: top-ranked model, all samples, chain pairs
    streamed = list(af3io.predictions.stream_predictions(alphafold3_examples_af3io, 'alphafold3_predictions/tetr_dimer_dna'))
    expected = list(af3io.predictions.stream_predictions(alphafold3_examples_dir / 'tetr_dimer_dna'))
    assert len(streamed) == len(expected) == 1
    for (name, model, pae, contact_probs, chain1, chain2), prediction in zip(streamed, expected):
        assert (name, model, chain1, chain2) == (prediction.name, prediction.model, None, None) == ('tetr_dimer_dna', prediction.model, None, None)
        np.testing.assert_array_equal(pae, prediction.pae)
        np.testing.assert_array_equal(contact_probs, prediction.contact_probs)
    samples = list(af3io.predictions.stream_predictions(alphafold3_examples_af3io, 'alphafold3_predictions/tetr_dimer_dna', top_model_only=False))
    assert [ sample.name for sample in samples ] == [ f'tetr_dimer_dna/seed-42_sample-{i}' for i in range(5) ]
    pairs = list(af3io.predictions.stream_predictions(alphafold3_examples_af3io, 'alphafold3_predictions/tetr_dimer_dna', chain_pairs=True))
    assert [ (pair.chain1, pair.chain2) for pair in pairs ] == [ (c1, c2) for c1 in 'ABCD' for c2 in 'ABCD' ]
    assert pairs[2].pae.shape == (218, 20)
    assert sum(1 for _ in af3io.predictions.stream_predictions(alphafold3_examples_af3io)) == 13

def test_jsonspan():
    from af3io.compress import jsonspan
    text = '{"a": [1, {"b": "x\\"y", "c": [2, 3]}], "d": {"b": null}, "e": "é"}'
    def select(path):
        return jsonspan.SPAN if path[-1:] == ('b',) else jsonspan.DESCEND
    spans = jsonspan.find(text, select)
    assert [ (path, text[start:end]) for path, start, end in spans ] == [ (('a', 1, 'b'), '"x\\"y"'), (('d', 'b'), 'null') ]
    assert jsonspan.find(text, lambda path: jsonspan.SKIP) == []
    for invalid in ['{"a": [1, 2', '{"a" 1}', '[1,]', '{"a": 1} x', '', '{"a": 1,}']:
        with pytest.raises(ValueError):
            jsonspan.find(invalid, lambda path: jsonspan.DESCEND)

def test_compress_data(alphafold3_examples_dir, alphafold3_examples_af3io, tmp_path):
    expected = decompressed_contents(alphafold3_examples_dir)
    data_names = [ name for name in expected if name.endswith('_data.json') ]
    with af3io.compress.open(alphafold3_examples_af3io) as reader:
        # MSAs and templates are stored once, e.g. TetR in tetr_homodimer, tetr_dimer_dna and tetr_dimer_tetracycline
        blobs = [ path for path, _ in reader.fs.walk('/blobs/data') ]
        n_refs = sum(reader.fs.read(f'/contents/data/{name}').count(b'af3io:///blobs/data/') for name in data_names)
        assert 0 < len(blobs) < n_refs
        assert { path.split('/')[3] for path in blobs } == {'unpairedMsa', 'pairedMsa', 'templates'}
        stub = json.loads(reader.fs.read(f'/contents/data/alphafold3_predictions/barnase_barstar/barnase_barstar_data.json'))
        assert stub['sequences'][0]['protein']['unpairedMsa'].startswith('af3io:///blobs/data/unpairedMsa/')
        for name in data_names:
            assert reader.read(name) == expected[name]
        assert reader.manifest['settings']['codecs']['data']['zstd_level'] == 19

    source = alphafold3_examples_dir / 'barnase_barstar'
    expected = decompressed_contents(source)
    for setting in ['store', 'drop']:
        image = tmp_path / f'{setting}.af3io'
        af3io.compress.compress([source], image, af3io.compress.Settings(data=setting), jobs=2)
        with af3io.compress.open(image) as reader:
            assert list(reader.fs.walk('/blobs/data')) == []
            if setting == 'store':
                assert reader.namelist() == sorted(expected)
                assert reader.fs.read('/contents/data/barnase_barstar/barnase_barstar_data.json') == expected['barnase_barstar/barnase_barstar_data.json']
            else:
                assert reader.namelist() == sorted(name for name in expected if not name.endswith('_data.json'))
                assert reader.manifest['stats']['dropped'] == { 'files': 1, 'bytes': len(expected['barnase_barstar/barnase_barstar_data.json']) }
        result = click.testing.CliRunner().invoke(af3io.cli.info, [str(image)])
        assert result.exit_code == 0 and ('dropped: 1 data pipeline JSONs' in result.output) == (setting == 'drop'), result.output
    with pytest.raises(ValueError):
        af3io.compress.Settings(data='zip')

def test_compress_fallback(tmp_path):
    # Files that cannot be compressed (as expected) are stored as-is under contents/other
    msa = '>query\\nMKTAYIAKQR' * 50
    files = {
        'pred/invalid_data.json': b'{"sequences": [ {"protein": {"unpairedMsa": "' + msa.encode() + b'" } } ',
        'pred/literal_data.json': json.dumps({ 'sequences': [ { 'protein': { 'unpairedMsa': msa, 'description': 'af3io:///blobs/data/x/ab/abc' } } ] }).encode(),
        'pred/ok_data.json': json.dumps({ 'sequences': [ { 'protein': { 'unpairedMsa': msa, 'pairedMsa': '' } } ] }, indent=2).encode(),
    }
    for name, data in files.items():
        (tmp_path / name).parent.mkdir(exist_ok=True)
        (tmp_path / name).write_bytes(data)
    for setting, fallbacks in [('compress', 2), ('store', 0)]:
        image = tmp_path / f'{setting}.af3io'
        af3io.compress.compress([tmp_path / 'pred'], image, af3io.compress.Settings(data=setting), jobs=1)
        with af3io.compress.open(image) as reader:
            assert dict(reader.iter()) == files
            assert reader.manifest['stats']['fallbacks'] == ({ 'data': fallbacks } if fallbacks > 0 else {})
            other = { path for path, _ in reader.fs.walk('/contents/other') }
            assert '/contents/other/pred/literal_data.json' in other # stored as-is, would be mis-read as a reference otherwise
            assert ('/contents/other/pred/invalid_data.json' in other) == (setting == 'compress')
            assert ('/contents/other/pred/ok_data.json' in other) == False
            if setting == 'compress':
                assert reader.fs.xattrs(reader.fs.lookup('/contents/other/pred/invalid_data.json'))[container.XATTR_FALLBACK] == b'data'
                assert b'af3io:///blobs/data/unpairedMsa/' in reader.fs.read('/contents/data/pred/ok_data.json')

def test_matrix():
    from af3io.compress import matrix
    rng = np.random.default_rng(0)
    chain_ids = ['A'] * 70 + ['B'] * 3 + ['C'] + ['D'] * 80
    values = rng.integers(0, 320, size=(154, 154)).astype(np.uint16)
    raw = matrix.format(values, 1)
    assert raw.startswith(b'[[') and b' ' not in raw
    assert json.loads(raw) == [ [ k / 10 for k in row ] for row in values.tolist() ]
    np.testing.assert_array_equal(matrix.parse(raw, 1), values)
    assert matrix.format(np.array([[0, 5, 10, 100]]), 2) == b'[[0.0,0.05,0.1,1.0]]'

    assert matrix.chain_groups(matrix.chain_runs(chain_ids), 64) == [('A-B-C', 0, 74), ('D', 74, 154)] # small chains join the previous group
    assert matrix.chain_groups(matrix.chain_runs(chain_ids), 1) == [('A', 0, 70), ('B', 70, 73), ('C', 73, 74), ('D', 74, 154)]
    assert matrix.chain_groups(matrix.chain_runs(['A'] * 3 + ['B/1'] * 2), 1) == [('0', 0, 3), ('1', 3, 5)]

    symmetric = np.triu(values) + np.triu(values, 1).T
    for data, is_symmetric in [(values, False), (symmetric, True)]:
        for ids in [chain_ids, None, ['A'] * 10]: # None or mismatched chain ids: a single block
            blob = matrix.encode(data, 1, ids, symmetric=is_symmetric)
            decoded, index = matrix.decode(blob)
            np.testing.assert_array_equal(decoded, data)
            assert index['decimals'] == 1 and index['symmetric'] == is_symmetric
        blob = matrix.encode(data, 1, chain_ids, symmetric=is_symmetric)
        with zipfile.ZipFile(io.BytesIO(blob)) as zf:
            assert sorted(zf.namelist()) == sorted(['matrix.json', 'A-B-C_A-B-C.jxl', 'A-B-C_D.jxl', 'D_D.jxl'] + ([] if is_symmetric else ['D_A-B-C.jxl']))
            assert all(info.compress_type == zipfile.ZIP_STORED for info in zf.infolist())
        ids = np.asarray(chain_ids)
        for chain1, chain2 in itertools.product('ABCD', repeat=2):
            np.testing.assert_array_equal(matrix.decode_block(blob, chain1, chain2), data[np.ix_(ids == chain1, ids == chain2)])
        with pytest.raises(KeyError):
            matrix.decode_block(blob, 'A', 'Z')
        assert matrix.encode(data, 1, chain_ids, symmetric=is_symmetric) == blob # deterministic

    for invalid in [b'[[1.0,2.0],[3.0]]', b'[[1.0,null]]', b'[[-1.0]]', b'[[]]', b'[1.0]', b'[[1.0] ]', b'[[1e9]]']:
        with pytest.raises(ValueError):
            matrix.parse(invalid, 1)
    with pytest.raises(ValueError):
        matrix.encode(np.zeros((2, 3), dtype=np.uint16), 1)

def confidences_json(pae, contact_probs, chain_ids):
    # As written by AlphaFold 3 (see af3io.confidences.dumps_json)
    return af3io.confidences.dumps_json(
        atom_chain_ids=chain_ids, atom_plddts=[ 50.0 ] * len(chain_ids), contact_probs=contact_probs, pae=pae,
        token_chain_ids=chain_ids, token_res_ids=list(range(1, len(chain_ids) + 1)),
    ).encode()

def test_compress_pae(alphafold3_examples_dir, alphafold3_examples_af3io, tmp_path):
    expected = decompressed_contents(alphafold3_examples_dir)
    names = [ name for name in expected if af3io.compress.classify(name) == 'confidences' ]
    with af3io.compress.open(alphafold3_examples_af3io) as reader:
        assert reader.manifest['stats']['fallbacks'] == {}
        n_blobs = sum(1 for _ in reader.fs.walk('/blobs/confidences/pae'))
        assert len(names) // 6 * 5 <= n_blobs < len(names) # top-ranked models are copies of samples
        for name in names:
            stub = json.loads(reader.fs.read(f'/contents/confidences/{name}'))
            assert stub['pae'].startswith('af3io:///blobs/confidences/pae/')
            js = reader.read_confidences(name)
            original = json.loads(expected[name])
            np.testing.assert_array_equal(js['pae'], np.array(original['pae']))
            assert js['token_chain_ids'] == original['token_chain_ids']
        df = reader.info()
        assert df.loc['pae', 'uncompressed'] > 10 * df.loc['pae', 'compressed'] > 0

    # Matrices that cannot be re-formatted exactly stay in the JSON
    chain_ids = ['A'] * 3 + ['B'] * 2
    pae = [ [ float(i + j) for j in range(5) ] for i in range(5) ]
    cp = [ [ 0.5 ] * 5 for _ in range(5) ]
    files = {
        'pred/ok_confidences.json': confidences_json(pae, cp, chain_ids),
        'pred/null_confidences.json': confidences_json([ row[:4] + [None] for row in pae ], cp, chain_ids),
        'pred/spaces_confidences.json': confidences_json(pae, cp, chain_ids).replace(b'[0.0,1.0', b'[0.0, 1.0'),
        'pred/invalid_confidences.json': confidences_json(pae, cp, chain_ids)[:-1],
    }
    for name, data in files.items():
        (tmp_path / name).parent.mkdir(exist_ok=True)
        (tmp_path / name).write_bytes(data)
    for setting in ['lossless', 'none']:
        image = tmp_path / f'{setting}.af3io'
        af3io.compress.compress([tmp_path / 'pred'], image, af3io.compress.Settings(pae=setting, contact_probs=False), jobs=1)
        with af3io.compress.open(image) as reader:
            assert dict(reader.iter()) == files
            stubs = { name: reader.fs.read(f'/contents/confidences/{name}') for name in files if reader.fs.lookup(f'/contents/confidences/{name}') }
            assert set(stubs) == set(files) - ({'pred/invalid_confidences.json'} if setting == 'lossless' else set())
            assert (b'af3io:///' in stubs['pred/ok_confidences.json']) == (setting == 'lossless')
            assert stubs['pred/null_confidences.json'] == files['pred/null_confidences.json']
            assert stubs['pred/spaces_confidences.json'] == files['pred/spaces_confidences.json']
            assert reader.manifest['stats']['fallbacks'] == ({ 'confidences': 1 } if setting == 'lossless' else {})

def test_compress_contact_probs(alphafold3_examples_dir, alphafold3_examples_af3io, tmp_path):
    from af3io.compress import matrix
    expected = decompressed_contents(alphafold3_examples_dir)
    names = [ name for name in expected if af3io.compress.classify(name) == 'confidences' ]
    with af3io.compress.open(alphafold3_examples_af3io) as reader:
        # One per seed: all samples (and the copy of the top-ranked one) of a seed have the same contact probabilities
        blobs = list(reader.fs.walk('/blobs/confidences/contact_probs'))
        assert len(blobs) == len(list(alphafold3_examples_dir.iterdir()))
        for _, inode in blobs:
            assert matrix.read_index(reader.fs.read(inode))['symmetric']
        for name in names:
            stub = json.loads(reader.fs.read(f'/contents/confidences/{name}'))
            assert stub['contact_probs'].startswith('af3io:///blobs/confidences/contact_probs/')
            np.testing.assert_array_equal(reader.read_confidences(name)['contact_probs'], np.array(json.loads(expected[name])['contact_probs']))
        df = reader.info()
        assert df.loc['contact_probs', 'uncompressed'] > 50 * df.loc['contact_probs', 'compressed'] > 0

    # Non-symmetric matrices are stored in full; --no-compress-contact-probs leaves them in the JSON
    chain_ids = ['A'] * 3 + ['B'] * 2
    pae = [ [ float(i + j) for j in range(5) ] for i in range(5) ]
    cp = [ [ round(0.01 * (i * 5 + j), 2) for j in range(5) ] for i in range(5) ]
    (tmp_path / 'pred').mkdir()
    (tmp_path / 'pred' / 'x_confidences.json').write_bytes(confidences_json(pae, cp, chain_ids))
    for flag in ['--compress-contact-probs', '--no-compress-contact-probs']:
        image = tmp_path / f'{flag}.af3io'
        result = click.testing.CliRunner().invoke(af3io.cli.compress, [flag, '-j', '1', str(tmp_path / 'pred'), str(image)])
        assert result.exit_code == 0, result.output
        with af3io.compress.open(image) as reader:
            assert reader.read('pred/x_confidences.json') == (tmp_path / 'pred' / 'x_confidences.json').read_bytes()
            blobs = list(reader.fs.walk('/blobs/confidences/contact_probs'))
            assert len(blobs) == (flag == '--compress-contact-probs')
            if len(blobs) > 0:
                assert not matrix.read_index(reader.fs.read(blobs[0][1]))['symmetric']
            np.testing.assert_array_equal(reader.read_confidences('pred/x_confidences.json')['contact_probs'], np.array(cp))
