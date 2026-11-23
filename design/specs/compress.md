# Compress/decompress
Compress [AlphaFold3 input/output](https://github.com/google-deepmind/alphafold3/tree/main/docs).
Implemented as `af3io` subcommands `compress`/`decompress`/`info`. By default, decompression reproduces every file name and file content bit-identically, like generic compression utilities such as `zip`/`unzip`.
With `--extract-archives` (the default), archives and compressed files (`.tar`, `.zip`, `.gz`, `.zst`) are unpacked before compression, so the guarantee applies to their unpacked contents: e.g. `confidences.json.zst` decompresses to a `confidences.json` that is bit-identical to the original after `zstd -d`. The archives themselves are not rebuilt.
Additional off-by-default lossy options allow for minor deviation in numerical accuracy but still keep exact formatting.

The compression format intentionally prefers to adapt existing tools and libraries to the context of the AlphaFold3 output.
For example, `contact_probs` is a [symmetric matrix](https://github.com/google-deepmind/alphafold3/issues/619) with limited discrete values that's identical across all samples with the same seed.
Therefore, can drop the lower triangle, compress using a standard lossless image format (e.g. JPEG XL lossless via `imagecodecs`), and rely on squashfs file deduplication to only save one copy per seed.
Another advantage of this approach is robustness to potential future [changes in the format](https://github.com/google-deepmind/alphafold3/blob/main/docs/input.md#versions).

## Implementation
Implement under a separate sub-package `af3io.compress`.

All files are kept in a squashfs container (.af3io extension), compressed with `zstd` and optimised compression settings. The container is organised as follows. For every input file at `<path>` there's a matching file under `/contents`, with different prefixes depending on the type:
- `/contents/confidences/<path>`
- `/contents/data/<path>`
- `/contents/model/<path>`
- `/contents/other/<path>` compresses the overall summary table (`ranking_scores.csv`), sample-specific summary confidenes JSON, AlphaFold3 weights license.

Specifically, data JSONs, confidences JSONs, and model CIFs take up the majority of space and have bespoke compression; all other files are added to `/contents/other/<path>` with squashfs compression and deduplication (e.g. license file is identical across all outputs).

Individual parts of data JSONs, confidences JSONs, and model CIFs (e.g. `contact_probs` matrices) are extracted from the text files, and compressed as custom binary blobs under the following hierarchy:

- `/blobs/confidences/contact_probs/<sha256>`
- `/blobs/confidences/pae/<sha256>`
- `/blobs/data/pairedMsa/<sha256>`
- `/blobs/data/templates/<sha256>`
- `/blobs/data/unpairedMsa/<sha256>`
- `/blobs/model/_atom_site/<sha256>`
- `/blobs/model/_entity_poly_seq/<sha256>`
- `/blobs/model/_ma_qa_metric_local/<sha256>`
- `/blobs/model/_pdbx_poly_seq_scheme/<sha256>`
- `/blobs/model/header/<sha256>`
- `/blobs/model/<category>/<sha256>`

Here `<sha256>` refers to the checksum of the decompressed content (exact replaced bytes); when storing on sqhuashfs, first two characters are used for a prefix directory like in git. This layout makes it easy to determine the total space after compression by file type. , e.g. for confidences files, can just sum up sizes for `/blobs/confidences` and `/contents/confidences`. Blob contents in the original file is replaced by `af3io://` followed by the absolute path of the blob in the container, e.g.:
```
af3io:///blobs/confidences/pae/58/58d1e17ffe5109a7ae296caafcadfdbe6a7d176f0bc4ab01e12a689b0499d8bd
```

For JSON files with compressed blobs, the absolute path is kept as a string wrapped in quotes such that the file stays a valid JSON

There's a manifest file at `/af3io_manifest.json` that contains the af3io version used to compress, as well as compression settings used. Any per-file metadata (created during compression by af3io) should be stored using xattrs

- Try to use squashfs command-line bindings e.g. mksquashfs, tar2sqfs or sqfstar
- Unrecognised files are assumed to be relevant post-processing results and added using squashfs default compression to `/contents/other/<path>`
- AlphaFold3 output naming patterns are documented in `af3io.predictions.Predictions`
- Minimal squashfs metadata - no permissions, user id, dates/times etc
- Use squashfs file attributes to store a hash of the original, uncompressed contents (and verify by default when uncompressing). For lossy compressors, the has is on lossy contents formatted as lossless contents (so verification still picks up data corruption)
- Compression should be as agnostic as possible to formatting changes between AlphaFold3 versions
- Every targeted method is verified per file at compression time (decompress and compare with the original); if the result is not identical, the file is stored as-is under /contents/other/<path> and counted as a fallback in `info`. That also covers future AlphaFold 3 versions and unusual inputs
- Every file is verified against the hash stored in the file attribute at decompression
- Can re-use the preliminary confidences compression implementations at af3io.confidences (and commands `confidences-compress` and `confidences-decompress`); use for details but remove the current half-disabled code
- Test correctness (round-trip) on `alphafold3_examples`, and `mgen_pools_5k_top3`

## Input-output options
- Basic syntax similar to `cp`; destination is the last positional argument always:
```
af3io compress af3io-testdata/mgen_pools_5k_top3 mgen_pools_5k_top3.af3io
af3io decompress mgen_pools_5k_top3.af3io . # decompresses to current directory
af3io decompress mgen_pools_5k_top3.af3io ./new_path # decompresses to ./new_path
af3io decompress mgen_pools_5k_top3.af3io 'pools_5k_0040f80/*' ./new_path # decompresses files matching pools_5k_0040f80/* to ./new_path
```
- ambiguous/undefined cases: try to mimic cp (e.g. multiple paths allowed under compress, last path is always the destination); use simpler option if matching cp adds excessive complexity (e.g. use python's fnmatch)
- error if destination image already exists
- `--extract-archives`/`--no-extract-archives` to avoid double-compression, decompress generic compression as already implemented for summary_confidences (.tar, .zip , .gz, .zst) before compression
    - default is enabled
    - contents of the archives is added in place of the archive
    - duplicates (input has both confidences.json and confidences.json.gz) is aborted with an error during compression
    - `decompress` does not re-build the archives, it decompresses as regular files (so e.g. confidences.json.gz becomes confidences.json)
- `--files-from` specifies a list of files-paths (one per line) to compress
    - each listed path is added at the root under its basename /x/y/pools_5k_0040f80 as pools_5k_0040f80/
    - `--files-from` can point to real files on the disk, not inside compressed files

## Compression options
- defaults are: `--compress-data`, `--compress-pae`, `--compress-contact-probs`, `--compress-model`
- `--lossy` is equivalent to: `--compress-data`, `--compress-pae-lossy`, `--compress-contact-probs`, `--compress-model`

### Data pipeline
- `--compress-data`/`--no-compress-data`/`--drop-data` whether to compress, add to squashfs (so compress with/by squashfs) or drop the data pipeline JSONs (`_data.json`) as they're often easy to re-create (keep by default)
    - see (re-use) current `data-fill` command for this 
    - compress is based on extracting `pairedMsa`, `unpairedMsa` and templates as these repeat across different JSONs containing the same sequence
    - these are then stored under (that way, data pipeline results are auto-compacted between files):
        - `/blobs/data/pairedMsa/<sha256>`
        - `/blobs/data/templates/<sha256>`
        - `/blobs/data/unpairedMsa/<sha256>`

### PAE matrix
- `--compress-pae`/`--no-compress-pae` compress PAE matrices in confidences JSON:
    - compressed PAE matrix is stored under `/blobs/confidences/pae/<sha256>`
    - pae field in confidences JSON under `/contents/confidences/<path>` is set to the location of the blob at `af3io:///blobs/confidences/pae/<sha256>`
    - PAE matrices are [partially symmetric](https://github.com/google-deepmind/alphafold3/issues/619)
    - values range from 0 to approx. 32 with one decimal digit
    - take advantage of the "block structure" defined by the chains in `token_chain_ids`; specifically, within-chain blocks (diagonal) have a different signal than between-chain blocks
        - alternativelly, define an even finer "block structure" by thresholding on (smoothed?) PAE values, e.g. something like 15A
        - possibly specialised compressors for on-diagonal vs off-diagonal blocks
    - adapt JPEG XL lossless via `imagecodecs` as the image codec, do optimise compression parameteres/filters
    - final format should, ideally, allow for fast random access to individual chain pair matrices; for example, uncompressed zip container of the per-block-pair jxls, e.g. A_B.jxl
    - exact confidences contact_probs/pae formatting (for round-tripping) is already implemented in af3io.confidences

### Contact probabilities matrix
- `--compress-contact-probs`/`--no-compress-contact-probs` compress contact_probs matrices in confidences JSON:
    - compressed contact_probs matrix is stored under `/blobs/confidences/contact_probs/<sha256>`
    - contact_probs field in confidences JSON under `/contents/confidences/<path>` is set to the location of the blob at `af3io:///blobs/confidences/contact_probs/<sha256>`
    - contact_probs is guaranteed symmetric
    - values are from 0 to 1 with two decimal digits
    - contact_probs is shared between all samples with a shared seed, such that squashfs deduplication will end up storing only one copy
    - exact confidences JSON formatting (for round-tripping) is already implemented in af3io.confidences

### Predicted structures (`--compress-model`)
`--compress-model`/`--no-compress-model` (default on) compress model CIFs (`*model.cif`) losslessly
- Each category (CIF split at #), apart from the data_ line, becomes a blob under `/blobs/model/<category>/<ab>/<sha256>`, named after its first item, e.g. `/blobs/model/_chem_comp/3f/3fa9...`. Text before the data_ line (the license header) goes in /blobs/model/header/
- In /contents/model/<path>, The # separators and the data_ line stay in the stub as they are. each section is replaced by its own reference line; sections below the size threshold stay inline e.g. af3io:///blobs/model/_chem_comp/3f/3fa9.... A line starting with af3io:// isn't valid CIF syntax, so it can't be confused with real content
- Four large tables stored as parquet files under:
    - `/blobs/model/_atom_site/<sha256>`
    - `/blobs/model/_entity_poly_seq/<sha256>`
    - `/blobs/model/_ma_qa_metric_local/<sha256>`
    - `/blobs/model/_pdbx_poly_seq_scheme/<sha256>`
- Parquet metadata may contain additional information to reproduce formatting exactly e.g. padding
- parquet column types chosen intelligently e.g. chains id-s are few discrete values so some kind of repeate encoding; coordinates with delta encoding etc
- Only externalise sections above a size threshold (e.g. 256 bytes feel free to adjust) and leave smaller ones inline.

### Lossy options
- `--compress-pae-lossy` discretise the whole PAE matrix before compressing:
    - explore thresholds actually used by various alphafold3 analysis methods in literature, write up in ADR please
    - proposed quantisation (feel free to explore alternatives):
        - exact for <= 15A (highest cutoff used by ipSAE/LIS/SPOC/domain tools)
        - after that in 2A bins for e.g. visualisation:
            (15, 17] -> 16A
            (17, 19] -> 18A
            (19, 21] -> 20A
            (21, 23] -> 22A
            (23, 25] -> 24A
            (25, 27] -> 26A
            (27, 29] -> 28A
            (29, ..) -> 31A
        - 31A for the final bin chosen to have a "round" number while also minimising mismatch for existing tools/scores such as iptm_from_pae
    - decompress all values to one decimal digit (match alphafold3 output formatting exactly)
    - calculate the hash on a string with original formatting but discretised values (so decompression hash check would still detect data corruption/formatting issues)

## Decompression/access
`af3io info` additional subcommand only shows uncompressed and compressed sizes by different types of files: data pipeline, models, pae, contact_probs, all other files, does not decompress

Implement/extend a `af3io.predictions.Predictions` to stream results from a compressed container, similar to e.g. zip files or tar files
- Only show "original file names" (`/contents` subpath) after decompression under a single hierarchy
- Do not decompress file on disk; instead, re-generate contents in RAM
- In addition to reading as text files, implement helpers similar to what's in predictions.py currently
- Ideally this would use one of the python modules for reading squashfs but use what's faster
- Two options:
    - either iterate over all files Predictions.iter(subpath='...')
    - iterate over predictions Predictions.stream_predictions(subpath='..', top_model_only=True/False, name, model, pae, contact_probs, chain_pairs=False)
    - for (name, model, pae, contact_probs, chain1, chain2) in p.stream_predictions('..')
    - add pairwise option as well?

## Performance tradeoffs
Max motivation/use case for developing this: 10% shards of roughly 300,000 pools (inputs of 5k residues) here: https://github.com/jurgjn/pooled-ppi-yeast
- Use this when balancing runtime-vs-size tradeoffs; ideally, should be able to compress all pools on a single HPC node with dozens of cores, 1-2 TB RAM
- 300,000 pools is 30TB zipped; af3io compress would ideally compress this into 10 shards
- "sharding" is manual, `compress --files-from` specifies a list of files to include on the image
- (note that deduplication doesn't cross shards..)

# Open questions
- Ideally, Predictions would open several "shards" as one file but leave for now
- Model coordinates could be compressed more aggressivelly using specialised compression like foldcomp; leave for now
