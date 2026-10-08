# Changelog
This project adheres to **YY.MINOR.MICRO**-style [Calendar Versioning](https://calver.org/).

## [26.1] - unreleased
- Require Python >= 3.13 (zstd support via built-in `compression.zstd` on 3.14, `backports.zstd` on 3.13)
- Added `fixname` to set the `name` attribute of input JSONs from their file names
- Added `summary-confidences` to collect summary confidences and interface scores from predictions (output directories or zip files) into a single parquet file
- Added `af3io.archive` to traverse and read files inside nested archives (tar, zip) with optional compression (gzip, zstd); predictions can now be read from inside other archives, e.g. `pools_5k.tar::pools_5k_0040f80.zip`
- Predictions can be read from output directories as-is, and from output directories or zip files with individually compressed files (gzip, zstd)
- Fixed `data-fill` dropping chain modifications (e.g. PTMs, modified bases) from input JSONs, and order chain fields as in data pipeline output (description last)
- `data-fill` matches protein/RNA chains by sequence as written by the data pipeline, i.e. taking modifications into account (`af3io.residue_names`, adapted from AlphaFold 3)
- Added interface scores to `summary-confidences`: ipSAE min (`chain_pair_ipsae10_min`, `chain_pair_ipsae15_min`), pairwise model confidence (0.8 ipTM + 0.2 pTM of the chain pair from PAE; `chain_pair_model_confidence`, `chain_pair_model_confidence_corrected` with size-corrected ipTM), cLIS, cLIA, iLIA, and interface size (`chain_pair_n_contacts`, `chain_pair_n_interface_residues`)
- LIS family (LIS, cLIS, iLIS, LIA, cLIA, iLIA) matches the AFM-LIS reference implementation exactly: PAE <= 12 (A) is confident (was < 12), and iLIS is the geometric mean of symmetrised LIS and cLIS (was the mean of per-direction iLIS)

## [26.0] - 2026-02-01
- Read inference results such as best structure or summary confidences from zip-compressed output
- Use calendar versioning as af3io is an
[amorphous set of utilities with research-driven scope](https://calver.org/#when-to-use-calver)

## [0.5] - 2026-01-21
- Fix `data-fill` missing sequence identification to account for all sequences across all input JSONs

## [0.4] - 2026-01-19
- Fix `data-fill` to only write one missing JSON for every input sequence that does not have data pipeline output

## [0.3] - 2026-01-14
- Fix `data-fill` crashing on data pipeline output with zero protein sequences

## [0.2] - 2026-01-12
- Specify `--data-dir` multiple times to use data pipeline output from multiple paths
- Added `--missing-dir` to create input JSON files for missing sequences

## [0.1] - 2026-01-05
- Added `af3io` command-line script with `data-fill`, `input-create`, `input-show` based on adhoc code from `jurgjn/batch-infer`
