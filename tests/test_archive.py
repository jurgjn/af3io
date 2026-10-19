import gzip, zipfile
try:
    from compression import zstd # Python >= 3.14
except ImportError:
    from backports import zstd
import pytest, af3io
from testdata import zip_bytes, tar_bytes

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
    inner_zip = zip_bytes(leaves)
    zip_of_zips = zip_bytes({ 'inner.zip': inner_zip, 'stored.zip': inner_zip }, compression=zipfile.ZIP_STORED)
    tar = tar_bytes({
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
        'tar_in_zip.zip': zip_bytes({ 'pools.tar': tar }),
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
