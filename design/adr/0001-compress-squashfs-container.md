# ADR 0001: squashfs container for `af3io compress`

- Status: Proposed
- Date: 2026-11-23
- Spec: [design/specs/compress.md](../specs/compress.md)

## Context

`af3io compress` stores AlphaFold 3 inputs/outputs in a single file that is a squashfs image (`.af3io`) compressed with zstd, with
file deduplication, per-file metadata in extended attributes (xattrs), and random access from Python without decompressing to disk.
This ADR covers the container itself (layout, writing, reading, settings, command-line interface); bespoke compression of data
pipeline JSONs, confidences JSONs, and model CIFs builds on it (later ADRs).

## Decision

### Layout

```
/af3io_manifest.json                     format version, af3io version, settings, totals (files, fallbacks, uncompressed bytes)
/contents/<kind>/<path>                  one file per input file; kind = data | confidences | model | other
/blobs/<kind>/<field>/<ab>/<sha256>      parts extracted from files under /contents (later ADRs)
```

- `kind` is derived from the file name (`af3io.compress.classify`): `*_data.json`, `*confidences.json` (not `*summary_confidences.json`),
  `*model.cif`, everything else is `other`. Files that fail verification after bespoke compression are stored as-is under
  `/contents/other/` (counted as fallbacks).
- Every file under `/contents` carries `user.af3io.sha256` and `user.af3io.size` of its original contents (plus
  `user.af3io.fallback=<kind>` for fallbacks). Reading a file always verifies the checksum (`af3io.compress.ChecksumError`).
- The manifest contains no timestamps, so identical inputs give bit-identical images (tested).

### Writing: `sqfstar` fed with a tar stream

The writer (`af3io.compress.container.Writer`) streams a PAX tar file into `sqfstar` (squashfs-tools >= 4.6), with xattrs as
`SCHILY.xattr.*` PAX records:

```
sqfstar -comp zstd -Xcompression-level 19 -b 1048576 -reproducible -mkfs-time 0 -all-time 0 -all-root -default-mode 0755 -quiet -no-progress
```

Alternatives considered:

- `mksquashfs` on a staging directory: needs every file (millions of blobs per shard) written to disk first, which is slow with
  file-count quotas on the cluster's parallel file systems, and needs `user.*` xattr support on the staging file system.
- `mksquashfs` with pseudo file definitions (`f` = file from the output of a command): one process per file.
- `tar2sqfs` (squashfs-tools-ng): similar to `sqfstar`, but a separate package; `sqfstar` comes with squashfs-tools.

The image is written to a temporary directory next to the destination and hard-linked into place at the end, so a failed run
leaves no partial image, and an existing destination is never overwritten.

### zstd level 19, 1 MiB blocks

`mksquashfs` on the uncompressed test data (64 cores, `-tailends`), time to read one confidences JSON with `unsquashfs -cat`:

| block size | zstd level | alphafold3_examples (206.5 MB) | mgen_pools_5k_top3 (5229.4 MB) | time (mgen) | read (mgen) |
|---|---|---|---|---|---|
| 128K | 3  | 43.0 MB | 415.1 MB | 2.4 s  | 0.09 s |
| 128K | 19 | 36.1 MB | 331.2 MB | 18.2 s | 0.10 s |
| 256K | 19 | 33.6 MB | 307.5 MB | 15.1 s | 0.09 s |
| 1M   | 3  | 36.8 MB | 371.7 MB | 2.4 s  | 0.08 s |
| 1M   | 9  | 32.8 MB | 339.5 MB | 2.5 s  | 0.10 s |
| 1M   | 15 | 31.6 MB | 316.3 MB | 3.6 s  | 0.08 s |
| 1M   | 19 | 28.6 MB | 271.1 MB | 13.1 s | 0.09 s |
| 1M   | 22 | 28.6 MB | 269.0 MB | 21.3 s | 0.08 s |

1 MiB (the largest squashfs block size) is consistently best; level 19 is within 1% of level 22 at ~60% of the time, and
decompression speed does not depend on the level. mksquashfs exposes no other zstd parameters (zstd's window is bounded by the
block size anyway). Tail ends of files are packed into fragments (`sqfstar` default).

For comparison, zip (deflate level 6, every file individually) gives 44.9 MB and 443.8 MB, so the container alone is ~1.6x
smaller than zip, see `paper/panels/compress_efficiency.pdf` (`tests/long/test_compress_long.py::test_compress_efficiency`).

### Reading: own minimal squashfs reader

`af3io.compress.squashfs` (~300 lines, standard library + zstd from `compression.zstd`/`backports.zstd`) reads the superblock,
inodes, directories, fragments and xattrs on demand, with caches for metadata blocks and directory listings. Latency of reading a
small file from a 28.6 MB image (20 files):

| reader | ms/file |
|---|---|
| `unsquashfs -cat` (one process per file) | 17.2 |
| PySquashfsImage (open + read) | 7.7 |
| `af3io.compress.squashfs` (open + read) | 2.2 |
| `af3io.compress.squashfs` (opened once) | 0.5 |

Alternatives considered:

- PySquashfsImage: no API to read xattrs (only to write them to extracted files), reads the whole directory tree when opening an
  image (cost grows with the image), and needs the `zstandard` package.
- `unsquashfs -cat`: a process per file that re-reads the image tables; no xattrs.
- squashfuse: needs FUSE, which is not available in containers or CI.

The reader also exposes the on-disk block sizes of every file, which `info` needs.

### Pipeline

- Inputs are enumerated with `af3io.archive.walk`: with `--extract-archives` (default), archives (`.tar`, `.zip`) and compressed
  files (`.gz`, `.zst`) are replaced by their contents in place, e.g. `dir/pred.zip` containing `pred/x.json.gz` becomes
  `dir/pred/x.json`. Duplicate names (e.g. `x.json` and `x.json.gz`) and unsafe names (absolute, `..`) are errors.
- Files are grouped by prediction directory (the parent directory, or the one above for `seed-*_sample-*/`), and groups are
  compressed in worker processes. Identical files within a group (e.g. the top-ranked model and its copy in `seed-*_sample-*/`)
  are compressed once. Results are written in input order, so the image does not depend on scheduling.
- Every blob is written once (squashfs would deduplicate it anyway, but this also skips the work).

### Command-line interface

- `af3io compress [--extract-archives/--no-extract-archives] [--files-from FILE] [-j N] SOURCES... DEST`: like `cp -r`, every
  source is added under its base name; `--files-from` adds paths (one per line, files or directories on disk) the same way. Error
  if `DEST` exists.
- `af3io decompress [--force] [-j N] IMAGE [PATTERNS...] DEST`: patterns use `fnmatch` (`*` also matches `/`); a pattern
  matching a directory selects everything below it (like `tar`). Unlike `cp`, existing files are not overwritten without
  `--force`, as decompressing a lossy container over its originals would silently replace them.
- `af3io info IMAGE`: uncompressed and compressed sizes by category (data, model, pae, contact_probs, rest of confidences JSONs,
  other files, squashfs metadata). Uncompressed sizes are totals recorded in the manifest at compression time; compressed sizes
  are summed from the block lists of all files, counting deduplicated data once and splitting fragment blocks in proportion to
  the tail ends they contain.

### Python interface

- `af3io.compress.open(path)` returns a `Container` with `namelist(subpath)`, `read(name)`, `open(name)`, `iter(subpath)`,
  `info()`; files are re-generated in memory, never on disk.
- Predictions in a container are addressed as `<image>.af3io::<path of the output directory>`, so `af3io.predictions.Predictions`,
  `find_predictions` and `af3io summary-confidences` work on containers as on directories and zip files.
- `af3io.predictions.stream_predictions(path, subpath, top_model_only, chain_pairs)` yields
  `(name, model, pae, contact_probs, chain1, chain2)` for any source (directories, archives, containers). The spec suggests
  `Predictions.iter`/`Predictions.stream_predictions`; as `Predictions` represents a single prediction, iteration over a container
  lives in `Container.iter` and the module-level `stream_predictions` instead.

## Consequences

- Compression needs squashfs-tools >= 4.6 (`sqfstar`, Ubuntu 24.04 or later); reading only needs Python.
- Empty directories, permissions and timestamps are not preserved.
- Members of compressed tar files (`.tar.gz`, `.tar.zst`) are re-read from the start of the archive, as elsewhere in af3io;
  prefer uncompressed tar files.
- Deduplication does not cross containers (shards).
