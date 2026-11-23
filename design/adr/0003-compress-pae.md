# ADR 0003: compressing PAE matrices

- Status: Proposed
- Date: 2026-11-23
- Spec: [design/specs/compress.md](../specs/compress.md), section "PAE matrix"
- Builds on: [ADR 0001](0001-compress-squashfs-container.md), [ADR 0002](0002-compress-data-pipeline.md)

## Context

The `pae` matrix (tokens x tokens, values from 0 to ~32 Å with one decimal) is the largest part of a confidences JSON: 128 MB of
text for a pool of 5,103 tokens, different for every sample. squashfs (zstd 19) compresses it ~26x; it is the largest part of
a container after `--compress-data` (see `paper/panels/compress_efficiency.pdf`).

## Decision

`--compress-pae` (default) stores the `pae` value of every confidences JSON as a blob `/blobs/confidences/pae/<ab>/<sha256>`
(`"pae": "af3io:///blobs/confidences/pae/..."` in the stub), see `af3io.compress.confidences` and `af3io.compress.matrix`:

- The matrix text is parsed into integers (values x 10) with `numpy.fromstring` (2.4 s for 5,103 x 5,103), and re-formatted
  with a lookup table of Python float representations (`repr(k / 10)`, as AlphaFold 3 writes them with `json.dumps`) in
  0.4 s. A matrix is only extracted if it re-formats to exactly the same bytes, otherwise it stays in the JSON as-is (e.g. if a
  future version of AlphaFold 3 writes two decimals, or nulls).
- The blob is an uncompressed zip file (`ZIP_STORED`, fixed timestamps) with an index (`matrix.json`: shape, decimals, runs of
  chains from `token_chain_ids`, chain groups) and one lossless JPEG XL image (`imagecodecs`, effort 7, 9 bits per sample) for
  every pair of chain groups, e.g. `A_B.jxl`. Groups are consecutive chains with at least 64 tokens together, so ions and
  single-atom ligand chains are merged with a neighbour (e.g. `A-B-C-D-E` for calmodulin with four calcium ions).
- `af3io.compress.matrix.decode_block(blob, chain1, chain2)` decodes a single image for one chain pair;
  `Container.read_confidences(name)` returns the matrices as numpy arrays without formatting them as text (used by
  `af3io.predictions`, e.g. `summary-confidences` and `stream_predictions`).

### Image codec and parameters

PAE matrices of the test data (one sample per prediction; mgen: 3 matrices of ~5k tokens, 384 MB of text; examples: 13
matrices, 4.4 MB):

| method | mgen | encode | examples |
|---|---|---|---|
| zstd 19 (uint16) | 14.92 MB | 52.9 s | 0.717 MB |
| PNG (16 bit, level 9) | 16.19 MB | 21.3 s | 0.653 MB |
| JPEG-LS (CharLS) | 10.88 MB | 0.6 s | 0.424 MB |
| JPEG XL effort 1 | 14.20 MB | 0.2 s | 0.502 MB |
| JPEG XL effort 3 | 10.54 MB | 3.0 s | 0.489 MB |
| JPEG XL effort 5 | 9.81 MB | 7.8 s | 0.418 MB |
| **JPEG XL effort 7** | **8.60 MB** | **11.1 s** | **0.380 MB** |
| JPEG XL effort 9 | 8.19 MB | 51.0 s | 0.372 MB |
| JPEG XL effort 7, 16 instead of 9 bits | 8.60 MB | 11.1 s | 0.381 MB |
| JPEG XL effort 7, symmetric residual (lower - upper^T) | 10.56 MB | 11.6 s | 0.429 MB |

Effort 7 is 12% smaller than effort 5 for 40% more time; effort 9 is another 5% for 4.6x the time. Decoding a 5,103 x 5,103 image
takes 1.2 s at effort 7 (single-threaded; 0.2 s with `decodingspeed=4`, which costs 27% in size), faster than parsing the
JSON text (2.4 s). Encoding PAE matrices of a pool of 5k tokens (5 samples) takes ~1 min of CPU with verification, i.e.
~8 hours for 30,000 pools on a node with 64 cores.

### Block structure

All samples of the test data (mgen: 15 matrices; examples: 65), including the zip file overhead:

| layout | mgen | examples |
|---|---|---|
| whole matrix | 42.81 MB | 1.908 MB |
| one image per chain (rows) | 42.74 MB | 1.939 MB |
| one image per chain pair | 41.65 MB | 1.994 MB |
| **one image per pair of chain groups of >= 64 tokens** | **41.60 MB** | 1.944 MB |
| one image per pair of chain groups of >= 256 tokens | 42.07 MB | 1.908 MB |
| chain groups split into tiles of <= 256 tokens | +3.6% | |
| chain groups split into tiles of <= 128 tokens | +15% | |
| rows of A_B interleaved with B_A^T (partial symmetry) | +19% | |

Blocks of chain pairs are 3% smaller than the whole matrix for pools (JPEG XL learns a model per image, and blocks on the
diagonal differ from those between chains), and allow reading a chain pair without decoding the rest. Merging small chains into
groups avoids the overhead of many tiny images (and zip entries, ~90 bytes each) for ions and ligands. Finer blocks (as a
stand-in for domains from thresholding PAE values) are worse, as are filters using the partial symmetry of PAE matrices.

### Results

`tests/long/test_compress_long.py::test_compress_efficiency` (`paper/panels/compress_efficiency.pdf`), PAE per prediction (in
earlier stages, PAE is stored within confidences JSONs, and its share is estimated):

| | AlphaFold 3 examples | mgen_pools_5k_top3 |
|---|---|---|
| uncompressed | 2.023 MB | 768.1 MB |
| zip | 0.463 MB | 53.8 MB |
| squashfs (zstd 19) | 0.272 MB | 29.0 MB |
| `--compress-pae` | 0.126 MB | 13.8 MB |

All files per prediction, `--drop-data --compress-pae`: 0.319 MB (examples) and 19.7 MB (pools), 10.8x and 7.5x smaller than zip.

## Consequences

- Needs `imagecodecs` (JPEG XL) for compressing and decompressing PAE matrices.
- The old, half-disabled `confidences-compress`/`confidences-decompress` (zarr) are removed, with the `zarr` dependency and the
  example notebook built on them.
