# Long tests (slow, or on larger data): marked `long` automatically, deselected by default, run with `pytest -m long`

from pathlib import Path
import pytest

def pytest_collection_modifyitems(config, items):
    long_dir = Path(__file__).parent
    for item in items:
        if item.path.is_relative_to(long_dir):
            item.add_marker(pytest.mark.long)
