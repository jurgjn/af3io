# Changelog

This project adheres to [Calendar Versioning](https://calver.org/).

## [26.0] - Unreleased
- Read inference results such as best structure or summary confidences from zip-compressed output
- Use **YY.MINOR.MICRO**-style calendar versioning as af3io is an
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
