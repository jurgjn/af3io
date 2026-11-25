[![CI](https://github.com/jurgjn/af3io/actions/workflows/python-package.yml/badge.svg)](https://github.com/jurgjn/af3io/actions/workflows/python-package.yml)
[![Coverage](https://raw.githubusercontent.com/jurgjn/af3io/main/.github/badges/coverage.svg)](https://github.com/jurgjn/af3io/actions/workflows/python-package.yml)
[![PyPI version](https://badge.fury.io/py/af3io.svg)](https://badge.fury.io/py/af3io)
[![Ask DeepWiki](https://img.shields.io/badge/Ask-DeepWiki-blue)](https://deepwiki.com/jurgjn/af3io)

# af3io
Collection of eclectic utilities for AlphaFold 3:
- Copy data pipeline output from existing results: [datafill.ipynb](examples/datafill/datafill.ipynb) ([open in Colab](https://colab.research.google.com/github/jurgjn/af3io/blob/main/examples/datafill/datafill.ipynb))
- Differences between input JSON files: [input_diff.ipynb](examples/input_diff/input_diff.ipynb) ([open in Colab](https://colab.research.google.com/github/jurgjn/af3io/blob/main/examples/input_diff/input_diff.ipynb))
- Set the `name` attribute of input JSONs to match their file names: `af3io fixname inputs/*.json` (use `--check` to only report mismatches)
- Collect summary confidences and interface scores (ipSAE, LIS, pDockQ, ...) from output directories or zip files, including predictions nested in other archives (tar, zip, gzip, zstd): `af3io summary-confidences predictions/ pools_5k.tar summary_confidences.parquet`
- Compress predictions into a single file (a squashfs image) with bit-identical decompression: `af3io compress predictions/ predictions.af3io.sqfs`, `af3io decompress predictions.af3io.sqfs .`, `af3io info predictions.af3io.sqfs` (needs `sqfstar` from squashfs-tools >= 4.6 to compress), or ~2.7x smaller for pooled predictions with `--lossy` (PAE values above 15 Å in 2 Å bins); predictions in it can be read directly, e.g. `af3io summary-confidences predictions.af3io.sqfs summary_confidences.parquet`
- Compress a prediction, read scores from the compressed file, and decompress it again: [confidences-compress.ipynb](examples/confidences-compress/confidences-compress.ipynb) ([open in Colab](https://colab.research.google.com/github/jurgjn/af3io/blob/main/examples/confidences-compress/confidences-compress.ipynb))
