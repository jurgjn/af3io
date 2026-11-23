# Long tests (slow, or on larger data): marked `long` automatically, deselected by default, run with `pytest -m long`

import datetime, subprocess
from pathlib import Path
import pytest

PAPER_DATA = Path(__file__).parents[2] / 'paper' / 'data'

def pytest_collection_modifyitems(config, items):
    long_dir = Path(__file__).parent
    for item in items:
        if item.path.is_relative_to(long_dir):
            item.add_marker(pytest.mark.long)

def _git_commit():
    # Commit of the working copy, with -dirty if there are uncommitted changes to tracked files
    cwd = Path(__file__).parent
    try:
        commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return 'unknown'
    return f'{commit}-dirty' if dirty else commit

@pytest.fixture
def record(request):
    """Save a table computed by a long test to paper/data/<test name>.parquet, tagged with the git commit and date
    (in the parquet metadata, see DataFrame.attrs), for paper panels to read (see design/specs/tests.md)
    """
    def record_(df):
        df = df.reset_index(drop=True)
        df.attrs.update(test=request.node.nodeid, git_commit=_git_commit(), date=datetime.date.today().isoformat())
        PAPER_DATA.mkdir(parents=True, exist_ok=True)
        df.to_parquet(PAPER_DATA / f'{request.node.name}.parquet')
        return df
    return record_
