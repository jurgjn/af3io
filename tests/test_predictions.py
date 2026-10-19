import functools, gzip, io, os, zipfile
try:
    from compression import zstd # Python >= 3.14
except ImportError:
    from backports import zstd
from pathlib import Path
import click, click.testing, pandas as pd, pytest, af3io, af3io.cli
from testdata import tar_bytes

@pytest.fixture(scope='session')
def example_predictions_tar(example_zip, tmp_path_factory):
    # Tar file with the example prediction twice: in a sub-directory, and compressed with zstd
    tar_path = tmp_path_factory.mktemp('af3io_tar') / 'pools.tar'
    data = example_zip.read_bytes()
    tar_path.write_bytes(tar_bytes({ 'dir/pools_5k_0040f80.zip': data, 'pools_5k_0040f80.zip.zst': zstd.compress(data) }))
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
