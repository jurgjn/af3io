try:
    from compression import zstd # Python >= 3.14
except ImportError:
    from backports import zstd
import click, click.testing, pytest, af3io, af3io.cli
from testdata import md5sum

@pytest.fixture(scope='session')
def example_confidences_json(example_predictions_dir, tmp_path_factory):
    path = tmp_path_factory.mktemp('af3io_confidences') / 'pools_5k_0040f80_confidences.json'
    path.write_bytes(zstd.decompress((example_predictions_dir / 'pools_5k_0040f80_confidences.json.zst').read_bytes()))
    return str(path)

'''
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
'''
