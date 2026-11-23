# ADR 0004: compressing contact probability matrices

- Status: Proposed
- Date: 2026-11-23
- Spec: [design/specs/compress.md](../specs/compress.md), section "Contact probabilities matrix"
- Builds on: [ADR 0003](0003-compress-pae.md)

## Context

The `contact_probs` matrix of a confidences JSON (tokens x tokens, values from 0 to 1 with two decimals) is 103 MB of text for a
pool of 5,103 tokens. It is symmetric, mostly zeros, and the same for all samples of a seed
([alphafold3#619](https://github.com/google-deepmind/alphafold3/issues/619)), so a prediction with one seed and five samples (plus
the copy of the top-ranked sample) has six identical copies.

## Decision

`--compress-contact-probs` (default) stores contact probabilities like PAE matrices ([ADR 0003](0003-compress-pae.md)): a blob
`/blobs/confidences/contact_probs/<ab>/<sha256>` with one lossless JPEG XL image (8 bits, effort 7) for every pair of chain groups,
and `"contact_probs": "af3io:///blobs/confidences/contact_probs/..."` in the stub.

- Symmetric matrices (checked for every matrix; all in the test data are) are stored as the blocks on and above the diagonal,
  with the lower triangle of diagonal blocks set to zero; other matrices are stored in full (`"symmetric": false` in the index).
- As blobs are named after the sha256 of the replaced text, the contact probabilities of a seed are stored once: workers skip
  encoding blobs already written (in the same prediction, or by other workers), and squashfs would deduplicate them anyway.
- Reading a single chain pair (`af3io.compress.matrix.decode_block`) transposes the stored block if it is below the diagonal.

### Parameters

One sample per prediction of the test data (mgen: 3 matrices of ~5k tokens, 309.7 MB of text; examples: 13 matrices, 4.2 MB):

| layout, JPEG XL effort | mgen | encode (mgen) | examples |
|---|---|---|---|
| full matrix, chain groups, effort 7 | 0.543 MB | 9.0 s | 0.071 MB |
| upper triangle, chain groups, effort 3 | 0.440 MB | 1.3 s | 0.069 MB |
| upper triangle, chain groups, effort 5 | 0.327 MB | 4.1 s | 0.048 MB |
| **upper triangle, chain groups, effort 7** | **0.309 MB** | **4.9 s** | **0.045 MB** |
| upper triangle, chain groups, effort 9 | 0.290 MB | 15.1 s | 0.040 MB |
| upper triangle, whole matrix, effort 7 | 0.279 MB | 7.2 s | 0.044 MB |

Dropping the lower triangle saves 43%. A single image for the whole matrix would be 10% smaller, i.e. ~10 KB per pool; the same
block structure as PAE matrices is kept for reading chain pairs without decoding everything, and for simplicity. Effort 9 saves
6% at three times the time.

### Results

`tests/long/test_compress_long.py::test_compress_efficiency` (`paper/panels/compress_efficiency.pdf`), contact probabilities per
prediction (in earlier stages, stored within confidences JSONs, and their share is estimated):

| | AlphaFold 3 examples | mgen_pools_5k_top3 |
|---|---|---|
| uncompressed | 1.960 MB | 619.5 MB |
| zip | 0.051 MB | 2.57 MB |
| squashfs (zstd 19) | 0.032 MB | 1.63 MB |
| `--compress-contact-probs` | 0.001 MB | 0.03 MB |

All files per prediction, `--drop-data --compress-pae --compress-contact-probs`: 0.305 MB (examples) and 18.2 MB (pools), 11.3x
and 8.1x smaller than zip.

## Consequences

- Contact probabilities shrink by ~1000x (310 MB of text to 0.3 MB per pool), so they no longer matter for the size of a
  container; their copies across samples cost nothing.
