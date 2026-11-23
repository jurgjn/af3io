# Shared helpers for tests: test data download/caching and in-memory archive construction

import gzip, hashlib, io, tarfile, zipfile
try:
    from compression import zstd # Python >= 3.14
except ImportError:
    from backports import zstd
from pathlib import Path
import pooch

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

def zip_bytes(members, compression=zipfile.ZIP_DEFLATED):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', compression=compression) as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return buf.getvalue()

def tar_bytes(members):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w') as tf:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()

def decompressed_contents(root):
    """{name: contents} of all files under root as added by af3io compress (under the base name of root, with .gz/.zst
    files decompressed), computed independently of af3io.compress
    """
    root = Path(root)
    contents = {}
    for path in sorted(root.rglob('*')):
        if path.is_file():
            name = path.relative_to(root.parent).as_posix()
            data = path.read_bytes()
            if name.endswith('.gz'):
                name, data = name.removesuffix('.gz'), gzip.decompress(data)
            elif name.endswith('.zst'):
                name, data = name.removesuffix('.zst'), zstd.decompress(data)
            contents[name] = data
    return contents

def read_dir(root):
    root = Path(root)
    return { path.relative_to(root).as_posix(): path.read_bytes() for path in sorted(root.rglob('*')) if path.is_file() }
