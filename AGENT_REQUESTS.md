# Agent requests

Missing tools/dependencies in the sandbox, for the human to install. Remove entries once resolved.

- **pytest + af3io test dependencies** (2026-10-19): the default Python (`/opt/venv`) has neither `pytest` nor `af3io` installed, so the test suite cannot run. Workaround used: a throwaway venv at `/tmp/af3io-venv` created with `uv venv` + `uv pip install -e '.[test]'`; on 2026-11-23 installed into `/opt/venv` with `uv pip install --python /opt/venv/bin/python -e '.[test]'` instead. Request: install `af3io` in editable mode with the `test` extra into the default environment.
- **imagecodecs** (2026-11-23): `af3io compress` stores PAE/contact probability matrices as JPEG XL images with `imagecodecs` (now a dependency in `pyproject.toml`), which is missing from the sandbox image. Workaround used: installed into `/opt/venv` with `uv pip install --python /opt/venv/bin/python imagecodecs`. Request: add `imagecodecs` (and drop `zarr`, no longer a dependency) in the Python packages of `.euvibe/sandbox.def`; `squashfs-tools` (>= 4.6, for `sqfstar`) is there already.
