
# pytest --capture=no --disable-warnings

import hashlib, os
import click, click.testing, pytest, af3io, af3io.cli

def md5sum(file):
    return hashlib.md5(open(file, 'rb').read()).hexdigest()

@pytest.fixture(scope='session')
def example_predictions_zip(tmp_path_factory):
    os.chdir(tmp_path_factory.mktemp('af3io_test'))
    os.system('curl -s https://zenodo.org/records/16920556/files/pools_5k.tar?download=1 | tar -xvf - --occurrence pools_5k_0040f80.zip')
    fn = os.path.join(os.getcwd(), 'pools_5k_0040f80.zip')
    #print(f'{os.path.getsize(fn)} bytes in {fn}')
    return fn

@pytest.fixture(scope='session')
def example_confidences_json(tmp_path_factory):
    os.chdir(tmp_path_factory.mktemp('af3io_test'))
    os.system('curl -s https://zenodo.org/records/16920556/files/pools_5k.tar?download=1 | tar -xvf - --occurrence pools_5k_0040f80.zip' + ' > /dev/null 2>&1')
    os.system('unzip -q pools_5k_0040f80.zip pools_5k_0040f80/pools_5k_0040f80_confidences.json')
    fn = os.path.join(os.getcwd(), 'pools_5k_0040f80/pools_5k_0040f80_confidences.json')
    #print(f'{os.path.getsize(fn)} bytes in {fn}')
    return fn

def test_confidences_compress_decompress(example_confidences_json):
    runner = click.testing.CliRunner()
    result_compress = runner.invoke(af3io.cli.confidences_compress, [example_confidences_json])
    assert result_compress.exit_code == 0
    #print(result_compress.output)

    # Calculate compression ratio based on file size, check that it's under a threshold..

    result_decompress = runner.invoke(af3io.cli.confidences_decompress, [example_confidences_json + '.af3io'])
    assert result_decompress.exit_code == 0
    #print(result_decompress.output)

    md5_downloaded = md5sum(example_confidences_json)
    md5_decompressed = md5sum(example_confidences_json + '.decompressed')
    assert md5_downloaded == md5_decompressed
