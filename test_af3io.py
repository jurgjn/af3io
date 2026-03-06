
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

@pytest.fixture(scope='session')
def example_data_json(tmp_path_factory):
    os.chdir(tmp_path_factory.mktemp('af3io_test_datafill'))
    os.system('curl -s https://zenodo.org/records/16920556/files/pools_5k.tar?download=1 | tar -xvf - --occurrence pools_5k_0040f80.zip' + ' > /dev/null 2>&1')
    os.system('unzip -q pools_5k_0040f80.zip pools_5k_0040f80/pools_5k_0040f80_data.json')
    fn = os.path.join(os.getcwd(), 'pools_5k_0040f80/pools_5k_0040f80_data.json')
    return fn

def test_data_fill(example_data_json):
    runner = click.testing.CliRunner()

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
    # af3io data-fill --data_dir pools_5k_0040f80 --input_dir monomer_jsons --output_dir monomer_msas
    os.makedirs('monomer_msas', exist_ok=True)
    result = runner.invoke(af3io.cli.data_fill, [
        '--data_dir', 'pools_5k_0040f80',
        '--input_dir', 'monomer_jsons',
        '--output_dir', 'monomer_msas'
    ])
    assert result.exit_code == 0

    # Create "index" mapping available sequences to data pipeline .jsons
    # af3io data-fill --data_dir monomer_msas --write-index
    result = runner.invoke(af3io.cli.data_fill, [
        '--data_dir', 'monomer_msas',
        '--write-index'
    ])
    assert result.exit_code == 0
    assert os.path.exists('monomer_msas/.af3io_data_index.json')

    # Create multimer input file with sequences from the original pool
    # The notebook uses a long command with all sequences.
    # Here we can just use the original file as base, but we want to test input-create too.
    args = ['multimer_jsons/pools_5k_0040f80.json', '--model_seed', '4']
    for i, seq in enumerate(sequences):
        chain_id = list(af3io.input.enumerate_chains())[i+1]
        args.extend(['--type', 'protein', '--id', chain_id, '--sequence', seq])

    os.makedirs('multimer_jsons', exist_ok=True)
    result = runner.invoke(af3io.cli.input_create, args)
    assert result.exit_code == 0

    # Then fill in data pipeline strings from the monomers
    # af3io data-fill --data_dir monomer_msas --json_path multimer_jsons/pools_5k_0040f80.json --output_dir multimer_msas/
    os.makedirs('multimer_msas', exist_ok=True)
    result = runner.invoke(af3io.cli.data_fill, [
        '--data_dir', 'monomer_msas',
        '--json_path', 'multimer_jsons/pools_5k_0040f80.json',
        '--output_dir', 'multimer_msas/'
    ])
    assert result.exit_code == 0

    # The result should be equal to what was downloaded from zenodo
    # Note: md5sum might differ if JSON formatting is slightly different, but af3io.input.write tries to be consistent.
    # The notebook shows they are identical.
    md5_original = md5sum(example_data_json)
    md5_filled = md5sum('multimer_msas/pools_5k_0040f80_data.json')

    # If md5 doesn't match due to key order or something, we might need a more robust check.
    # But let's try MD5 first as the notebook suggests it works.
    assert md5_original == md5_filled
