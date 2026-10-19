## Tests/performance

- Use test data from: https://github.com/jurgjn/af3io-testdata
- Test data from elsewhere: immutable sources (e.g. a Zenodo record) may be used directly. For mutable sources (e.g. other GitHub repositories), don't fetch from them directly; add an entry to `AGENT_REQUESTS.md` for the human to fork the repository (or copy the files into af3io-testdata).
- Fetch test data with pooch through `tests/testdata.py`: pinned to a commit of af3io-testdata, verified by sha256 in `tests/test_data_registry.txt`, and cached locally (override the cache location with `$AF3IO_TEST_DATA`).
- Short tests live in `tests/` and run in CI with plain `pytest`. Each should take at most a few minutes once its data is cached.
- Long tests (slow, or on larger data) live in `tests/long/`. They are marked `long` automatically, deselected by default, and only run manually with `pytest -m long`. They never run in CI.
- Fixtures shared by short and long tests go in `tests/conftest.py`. Fixtures used only by long tests go in `tests/long/conftest.py`.
- A long test that a paper panel depends on first computes its table, then saves it with the `record` fixture to `paper/data/<test name>.parquet` (tagged with git commit and date), then asserts on it. Panels only read these files and never recompute them. `make -C paper data` re-runs the long tests to refresh the tables.