# ADR 0002: compressing data pipeline JSONs

- Status: Proposed
- Date: 2026-11-23
- Spec: [design/specs/compress.md](../specs/compress.md), section "Data pipeline"
- Builds on: [ADR 0001](0001-compress-squashfs-container.md)

## Context

Data pipeline output (`*_data.json`) is the input JSON with MSAs (`unpairedMsa`, `pairedMsa`) and templates for every protein/RNA
chain. These are the bulk of the file (e.g. 330 MB for a pool of 18 proteins), and identical for the same sequence across
predictions: in the motivating use case (pools of ~5k residues from a few thousand yeast proteins), every sequence is part of
many pools.

## Decision

`--compress-data` (default) replaces the value of every `sequences[i].<type>.{unpairedMsa,pairedMsa,templates}` of at least
256 bytes by a reference to a blob, `/blobs/data/<field>/<ab>/<sha256>`:

```json
"unpairedMsa": "af3io:///blobs/data/unpairedMsa/3f/3fa9...",
```

- The blob is the exact JSON text of the value (a quoted, escaped string, or the array of templates with its indentation),
  i.e. `json.loads(blob)` gives the MSA. Splicing exact bytes back keeps decompression independent of JSON formatting
  (indentation, escaping, key order, versions of AlphaFold 3), and the `sha256` (blob name) is that of the replaced bytes, as in
  the spec.
- Values are located with a small JSON walker (`af3io.compress.jsonspan`) that descends only into `sequences[i].<type>` and
  skips everything else with the C scanner of the `json` module (0.4 s for a 330 MB data JSON).
- Blobs are compressed with zstd level 19 and a long window (`window_log=27`, long distance matching), before squashfs, which
  then stores them as they are. On all MSAs/templates of the test data (unique values):

  | | uncompressed | squashfs (zstd 19, 1 MiB blocks) | zstd 19, whole value | zstd 22, whole value |
  |---|---|---|---|---|
  | unpairedMsa | 212.9 MB | 45.8 MB | 40.1 MB | 40.0 MB |
  | pairedMsa | 594.5 MB | 88.1 MB | 68.1 MB | 67.7 MB |
  | templates | 37.0 MB | 5.8 MB | 5.6 MB | 5.6 MB |

  Compressing a whole MSA with a window that spans it is 19% smaller than squashfs' independent 1 MiB blocks; level 22 is not
  worth its time. Decompression needs at most 128 MiB of window, the zstd default limit.
- `--no-compress-data` stores data JSONs as-is (squashfs compression only); `--drop-data` leaves them out of the container
  (counted in the manifest as `dropped` and shown by `info`), as they are often easy to re-create.

### Changes to the pipeline (all codecs)

- Codecs (`af3io.compress.codec`) split a file into a stub and spans (cheap), and encode each span into a blob (expensive).
  Each blob is verified to decode to its span, and the stub to re-generate the original file from the spans; otherwise the
  file is stored as-is under `/contents/other/` and counted as a fallback (e.g. invalid JSON).
- Blobs written already are skipped before encoding: within a group of files (e.g. contact probabilities shared by all
  samples of a seed), and across worker processes through a shared set of written blob paths. Encoding an MSA takes seconds,
  and the same MSAs come up in many pools.
- Files of a kind with a codec that are stored as-is (e.g. `--no-compress-data`) but contain text that would be read as a
  reference (`af3io:///blobs/...`) are stored under `/contents/other/`.
- squashfs packs small files and tail ends into shared fragment blocks in the order they are written, so the writer emits the
  files of a prediction grouped by kind (e.g. the models of all samples next to each other), then the new blobs. On the
  AlphaFold 3 examples this is 1% smaller than writing in path order. As blocks are shared, `info` splits their compressed size
  in proportion to uncompressed bytes, so sizes by category are approximate for small files (the total is exact).

### Results

`tests/long/test_compress_long.py::test_compress_efficiency` (`paper/panels/compress_efficiency.pdf`), sizes per prediction:

| | AlphaFold 3 examples | mgen_pools_5k_top3 |
|---|---|---|
| zip | 3.45 MB | 147.9 MB |
| squashfs only (`--no-compress-data`) | 2.19 MB | 90.4 MB |
| `--compress-data` | 1.54 MB | 68.2 MB |
| `--drop-data` | 0.45 MB | 35.0 MB |

The test data has little overlap between predictions (155 unique of 188 MSAs/templates); with thousands of pools per container,
most MSAs are stored once.

## Alternatives considered

- Storing the decoded MSA (a3m) instead of the JSON string: blobs would be directly usable as files, but decompression would
  depend on re-creating AlphaFold 3's JSON string escaping; `json.loads(blob)` is a simple way to get the MSA.
- Compressing `pairedMsa` with the `unpairedMsa` of the same chain as a zstd dictionary: 17% smaller MSAs of a pool (38.1 vs
  45.8 MB), but blobs would depend on each other. As MSAs are stored once per sequence in a container (~15 GB for a few
  thousand yeast proteins vs. ~600 GB for a shard of 30,000 pools), this saves <0.5%; left for later.
- `af3io data-fill` (re-creating data JSONs from input JSONs and an index of data JSONs): needs the input JSONs and does not
  reproduce files bit-identically by construction; blob references keep the data JSON self-contained.
