# Compression efficiency (long test)

Runs compress-decompress cycle, cumulatively adding more aggressive compression settings:
- Uncompressed input (regardless of whether e.g. compressed on disk)
- All files zipped (baseline)
- `af3io compress` i.e. just build a squashfs container (so explicitly disable all compression except squashfs built-in zstd)
- `--compress-data`
- `--drop-data`
- `--compress-pae`
- `--compress-contact-probs`
- `--compress-model`
- `--compress-pae-lossy`

Test runs compression on all options, checks that compression was successful by decompressing, writes sizes data into a summary, then another script to load this data and visualise:
- Horizontal bar chart, size of the uncompressed files (on a broken x axis), then .zip-compressed files, then total size after cumulatively adding all "tricks"
- Bars on the bar chart are colored by the amount of space taken by (left to right) all remaining data, models, contact_probs, pae, data pipeline output (`*_data.json`)
- At the end of each bar plot a label with total size; for anything smaller then zip add how many times smaller than zip
- Run separately `alphafold3_examples/` and  `mgen_pools_5k_top3/`, normalise all sizes by number of inputs (e.g. size per pooled prediction)
