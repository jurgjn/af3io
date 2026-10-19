import os
import click, click.testing, af3io, af3io.cli
from testdata import md5sum

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
