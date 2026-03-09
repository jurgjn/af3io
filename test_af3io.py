
# pytest --capture=no --disable-warnings

import hashlib, os
import click, click.testing, pytest, af3io, af3io.cli

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
