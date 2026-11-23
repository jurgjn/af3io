# ADR 0006: lossy compression of PAE matrices

- Status: Proposed
- Date: 2026-11-23
- Spec: [design/specs/compress.md](../specs/compress.md), section "Lossy options"; tests:
  [test_compress_pae_lossy_accuracy_long.md](../specs/test_compress_pae_lossy_accuracy_long.md),
  [test_compress_pae_lossy_bins_long.md](../specs/test_compress_pae_lossy_bins_long.md)
- Builds on: [ADR 0003](0003-compress-pae.md)

## Context

After lossless compression ([ADR 0003](0003-compress-pae.md)), PAE matrices are most of a container of pooled predictions
(13.8 of 16.1 MB per pool). In pools of ~5k tokens, most pairs of proteins do not interact: 95% of PAE values are above 15 Å and
92% above 29 Å, where values only say "no confident relative position" (`paper/panels/compress_pae_lossy_bins.pdf`):

| PAE | AlphaFold 3 examples | mgen_pools_5k_top3 |
|---|---|---|
| <= 15 Å | 88.7% | 4.7% |
| (15, 29] Å | 9.2% | 3.7% |
| > 29 Å | 2.1% | 91.7% |
| mean of values > 29 Å | 30.27 Å | 31.48 Å |
| largest value | 31.5 Å | 31.7 Å |

### PAE thresholds of analysis methods

From the papers in the Paperpile library (citekeys) and the reference implementations used by `af3io.scores`:

| method | use of PAE | threshold | source |
|---|---|---|---|
| LIS, LIA, cLIS, cLIA, iLIS (AFM-LIS) | mean of 1 - PAE/12 (or count) over inter-chain pairs with PAE <= 12 Å; the cutoff maximised ROC AUC among cutoffs from 1 to 30 Å | 12 Å | `Kim2024-li`, also FlyPredictome `Kim2026-vz` |
| ipSAE | pTM-like score over residue pairs with PAE below a cutoff, d0 from the number of such residues | 10 Å (default), 15 Å; 12 Å in `Badonyi2026-iy` | `Dunbrack2025-fo` |
| SPOC (Predictomes) | interface residue pairs with min(PAE(x, y), PAE(y, x)) < 15 Å (and pLDDT > 50, distance < 5 Å) | 15 Å | `Schmid2025-mp`, `Elofsson2025-nt` |
| DPAM (domain parser) | residues with PAE < 8 Å are in the same domain with probability >= 50%; "PAE neighbours" below 6 Å (4 to 12 Å tested) | <= 12 Å | `Zhang2023-ow` |
| ipAE (success criterion in the actifpTM paper) | median PAE of interface residues within 8 Å below 5 Å | 5 Å | `Varga2025-uj` |
| ipTM from PAE (AlphaFold-Multimer), pTM | pTM-like transform 1/(1 + (PAE/d0)^2) of every inter-chain (or every) pair | none | `Evans2022-al` |
| actifpTM, reactifpTM, actifpSAE | pTM-like transform of every pair, weighted by contact probabilities or contacts | none | `Varga2025-uj`, `Simpkin2026-sd` |
| pDockQ2 | mean of 1/(1 + (PAE/10)^2) over interface contacts (8 Å) | none (d0 = 10 Å) | `Zhu2023-fp` |
| Pinc | probability of contact (12 Å radius) with PAE as the uncertainty of positions | none | `Badonyi2026-iy` |

All thresholds are at most 15 Å, so keeping values up to 15 Å exactly keeps every threshold-based score exactly. Scores that
transform every PAE value (pTM-like, pDockQ2, Pinc) change slightly, as values above 15 Å contribute little: e.g.
1/(1 + (PAE/d0)^2) for d0 = 10 Å is 0.31 at 15 Å and 0.09 at 31 Å.

## Decision

`--compress-pae-lossy` (and `--lossy`, the same as `--compress-data --compress-pae-lossy --compress-contact-probs
--compress-model`) quantises PAE matrices before compressing them as in [ADR 0003](0003-compress-pae.md):

- values up to 15 Å (the highest cutoff above) are kept exactly;
- larger values are put into 2 Å bins, (15, 17] -> 16, (17, 19] -> 18, ..., (27, 29] -> 28, and all values above 29 Å into a
  final bin -> 31 Å (`af3io.compress.confidences.PAE_LOSSY_BINS`, recorded in the manifest);
- the matrix is written with the formatting of the original (one decimal, as AlphaFold 3 writes it), so a decompressed file
  differs from the original only in PAE values above 15 Å, and has the same size. The blob is named after, and the checksum
  of the file (`user.af3io.sha256`) computed on, the file with quantised values, so verification at decompression still
  detects corruption and formatting problems. Compression verifies that the original re-formats exactly (otherwise the matrix
  stays as-is), and that the blob decodes to the quantised text.
- `af3io decompress` notes when PAE matrices were compressed lossily.

### Quantisation

All samples of the test data (mgen: 15 matrices; examples: 65), size of the PAE blobs and differences of ipTM computed from
PAE (`af3io.scores.iptm_from_pae`, all pairs of different chains) to lossless PAE:

| quantisation above 15 Å | mgen size | mgen ipTM from PAE, mean / max difference | examples size |
|---|---|---|---|
| none (lossless) | 41.62 MB | | 1.963 MB |
| 1 Å bins, final bin 31 Å | 11.95 MB | +0.0015 / 0.0055 | 1.791 MB |
| **2 Å bins, final bin 31 Å** | **11.26 MB** | **+0.0015 / 0.0054** | **1.770 MB** |
| 2 Å bins, final bin 30 Å | | +0.0057 / 0.0134 | |
| 2 Å bins, final bin 30.5 Å | | +0.0035 / 0.0092 | |
| 2 Å bins, final bin 31.5 Å | | -0.0005 / 0.0084 | |
| 4 Å bins, final bin 30 Å | 10.43 MB | +0.0056 / 0.0134 | 1.745 MB |

Lossy PAE matrices of pools are 3.7x smaller than lossless ones; for the examples, which mostly have confident interfaces, the
gain is small (10%). 2 Å bins are close in size to 4 Å bins with less than half the differences. Of the values tried for the
final bin, 31 Å has the smallest largest difference (31.5 Å, close to the mean of values above 29 Å in pools, has a smaller mean
difference but a larger largest difference), and is a round number.

For a single pair, a pTM-like term 1/(1 + (PAE/d0)^2) changes by at most 0.0067, 0.0192, 0.0240, 0.0274 for d0 = 4, 8, 10, 12 Å
(at a value of 15.1 Å, which becomes 16 Å, or 17.0 Å); averages over many pairs change much less.

### Accuracy of scores

`tests/long/test_compress_pae_lossy_long.py::test_compress_pae_lossy_accuracy` (`paper/panels/compress_pae_lossy_accuracy.pdf`)
runs `af3io summary-confidences` on containers with lossless and lossy PAE (both datasets, all samples, every pair of chains):

- the same: all threshold-based scores (ipSAE 10/15 and their minimum, LIS, LIA, cLIS, cLIA, iLIS, iLIA), and all scores not
  using PAE (ipTM, pDockQ, ipLDDT, contacts, contact probabilities);
- different (largest absolute difference, mean difference, and correlation for the pools, where differences are largest; the
  test requires the bounds in parentheses and r > 0.9999 over both datasets):

| score | max abs difference | mean difference | r |
|---|---|---|---|
| reactifpTM (0.03) | 0.0154 | -0.00006 | 0.99998 |
| actifpTM from PAE (0.03) | 0.0152 | +0.0007 | 0.99997 |
| Pinc (0.02) | 0.0090 | -0.00003 | 0.99998 |
| ipTM from PAE (0.01) | 0.0049 | +0.0012 | 0.99996 |
| model confidence, corrected (0.002) | 0.0007 | +0.0001 | 0.999999 |
| actifpSAE (0.001) | 0.0003 | +0.00001 | 1.0 |
| pDockQ2 (0.001) | 0.0003 | -0.000002 | 1.0 |

### Results

`tests/long/test_compress_long.py::test_compress_efficiency` (`paper/panels/compress_efficiency.pdf`), per prediction with all
other options (`--drop-data --compress-contact-probs --compress-model`):

| | AlphaFold 3 examples | mgen_pools_5k_top3 |
|---|---|---|
| PAE, `--compress-pae` | 0.131 MB | 13.80 MB |
| PAE, `--compress-pae-lossy` | 0.116 MB | 3.60 MB |
| all files, `--compress-pae` | 0.245 MB (14.1x smaller than zip) | 16.08 MB (9.2x) |
| all files, `--compress-pae-lossy` | 0.230 MB (15.0x) | 5.88 MB (25.2x) |

For the motivating use case (300,000 pools, 30 TB as zip files), this is ~1.8 TB without data pipeline JSONs (~4.8 TB with
lossless PAE).

## Consequences

- Lossy containers re-generate confidences JSONs that differ from the originals in PAE values above 15 Å; everything else
  (including all other files) is bit-identical, and every file is verified against its checksum.
- Scores with PAE cutoffs up to 15 Å are unaffected; scores using all PAE values change by up to ~0.015 on the test data.
