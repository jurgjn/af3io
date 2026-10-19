# Changelog
This project adheres to **YY.M.MICRO**-style [Calendar Versioning](https://calver.org/), e.g. 26.10 or 26.10.1. 

## [26.10] - unreleased
- Added reading predictions from output directories as-is, and from output directories or zip files with individually compressed files (gzip, zstd)
- Added `summary-confidences` to collect summary confidences and interface scores from predictions (output directories or zip files) into a single parquet file
- Added `af3io.archive` to traverse and read files inside nested archives (tar, zip) with optional compression (gzip, zstd); predictions can now be read from inside other archives, e.g. `pools_5k.tar::pools_5k_0040f80.zip`
- Added interface scores per chain pair (`af3io.scores`) to `summary-confidences`
- Added tests of interface scores against reference implementations
- Added `data-dump` to extract MSAs (`.a3m`) and templates (`.cif`) from data pipeline output into separate files
- Added `confidences-show` to show a compact summary of a confidences JSON
- Added `fixname` to set the `name` attribute of input JSONs from their file names
- Added publishing of tagged releases to PyPI
- Changed `data-fill` to match protein/RNA chains by sequence as written by the data pipeline, i.e. taking modifications into account (`af3io.residue_names`, adapted from AlphaFold 3)
- Changed minimum Python version to 3.13 (zstd support via built-in `compression.zstd` on 3.14, `backports.zstd` on 3.13)
- Fixed `data-fill` dropping chain modifications (e.g. PTMs, modified bases) from input JSONs, and chain fields ordered differently from data pipeline output (description last)
- Fixed `data-fill` failing on inputs with ligands (#3) or DNA chains (#4)

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
