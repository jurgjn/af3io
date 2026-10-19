# pytest tests/ --capture=no --disable-warnings

import zipfile
try:
    from compression import zstd # Python >= 3.14
except ImportError:
    from backports import zstd
from pathlib import PurePosixPath
import pytest
from testdata import fetch_test_dir

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
