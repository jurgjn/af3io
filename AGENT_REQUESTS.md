# Agent requests

Missing tools/dependencies in the sandbox, for the human to install. Remove entries once resolved.

- **pytest + af3io test dependencies** (2026-10-19): the default Python (`/opt/venv`) has neither `pytest` nor `af3io` installed, so the test suite cannot run. Workaround used: a throwaway venv at `/tmp/af3io-venv` created with `uv venv` + `uv pip install -e '.[test]'`. Request: install `af3io` in editable mode with the `test` extra into the default environment.
