# ADR 0005: compressing predicted structures (model CIFs)

- Status: Proposed
- Date: 2026-11-23
- Spec: [design/specs/compress.md](../specs/compress.md), section "Predicted structures"
- Builds on: [ADR 0001](0001-compress-squashfs-container.md), [ADR 0002](0002-compress-data-pipeline.md)

## Context

Model CIFs (`*model.cif`, 3.6 MB for a pool of 41k atoms) are the third largest part of a prediction after confidences and data
pipeline JSONs. AlphaFold 3 writes them as mmCIF/ModelCIF with one category per section (sections separated by `#` lines), and
pads the values of every loop to the width of the longest value in its column, e.g.

```
ATOM 1    N N   . ALA A 1 1   ? 0.932   -20.839 19.235  1.00 68.19 1   A 1 
```

In all 96 model CIFs of the test data, four loops make up 99.1% of the bytes (`_atom_site` 89.5%, `_pdbx_poly_seq_scheme` 4.0%,
`_ma_qa_metric_local` 3.6%, `_entity_poly_seq` 2.0%); every loop of these four follows the padding rule exactly (each value
left-justified to its column width and followed by a space).

## Decision

`--compress-model` (default) splits every model CIF at `#` lines (`af3io.compress.model`):

- Text before the `data_` line (the terms of use) becomes a blob `/blobs/model/header/<ab>/<sha256>`; the `data_` line and the
  `#` lines stay in the stub.
- Every other section of at least 256 bytes becomes a blob `/blobs/model/<category>/<ab>/<sha256>`, named after its first
  item (e.g. `_chem_comp`), and is replaced in the stub by a reference line `af3io:///blobs/model/<category>/<ab>/<sha256>`
  (not valid CIF, so it cannot be confused with contents). Smaller sections stay inline. Sections that are the same in
  several models (e.g. `_chem_comp`, `_pdbx_data_usage`, the header, and everything in the copy of the top-ranked model) are
  stored once.
- Loops of `_atom_site`, `_entity_poly_seq`, `_ma_qa_metric_local`, `_pdbx_poly_seq_scheme` are stored as parquet files; all
  other sections (and loops of these categories that do not re-format exactly) as text, compressed by squashfs.

### Parquet tables

- Column types are chosen per column from the values: integers (with `.` or `?` as nulls), fixed-point decimals stored as
  integers (e.g. coordinates x1000), or strings. A numeric type is only used if formatting the numbers gives back exactly the
  same values (e.g. `-0.000` or `007` stay strings).
- Integer columns are written with `DELTA_BINARY_PACKED` (consecutive atoms are ~1.5 Å apart, ids increase by 1), string columns
  with dictionary encoding, everything compressed with zstd level 19 inside parquet.
- The header lines (`loop_`, item names) and the column widths are stored in the parquet metadata (`af3io` key), so a loop is
  re-generated exactly; the result is verified when compressing.

Sizes of the four loops in one sample of each pool of `mgen_pools_5k_top3` (3 x ~41k atoms):

| encoding | `_atom_site` | `_ma_qa_metric_local` | `_pdbx_poly_seq_scheme` | `_entity_poly_seq` |
|---|---|---|---|---|
| text | 9960.5 KB | 401.7 KB | 442.5 KB | 218.6 KB |
| text, zstd 19 | 1768.0 KB | 99.6 KB | 70.4 KB | 36.5 KB |
| parquet, dictionary encoding (default), zstd 19 | 2204.3 KB | 109.2 KB | 57.3 KB | 26.4 KB |
| parquet, plain encoding of numbers, zstd 19 | 1506.1 KB | 85.1 KB | 31.0 KB | 17.6 KB |
| parquet, byte stream split of numbers, zstd 19 | 1013.8 KB | 39.7 KB | 20.1 KB | 13.9 KB |
| **parquet, delta encoding of numbers, zstd 19** | **941.1 KB** | **38.2 KB** | **19.1 KB** | **13.6 KB** |
| parquet, delta encoding of numbers, zstd 3 | 959.1 KB | 38.4 KB | 19.6 KB | 13.8 KB |

With delta encoding, `_atom_site` is 1.9x smaller than zstd on the text; coordinates take ~14 bits each, and per-atom pLDDT
(`B_iso_or_equiv`) ~12 bits, while all other columns are almost free (dictionary/run-length encoded). Encoding all 96 model CIFs
of the test data takes 9.2 s, decoding 2.7 s.

### Results

`tests/long/test_compress_long.py::test_compress_efficiency` (`paper/panels/compress_efficiency.pdf`), models per prediction:

| | AlphaFold 3 examples | mgen_pools_5k_top3 |
|---|---|---|
| uncompressed | 0.981 MB | 21.95 MB |
| zip | 0.260 MB | 5.56 MB |
| squashfs (zstd 19) | 0.130 MB | 3.87 MB |
| `--compress-model` | 0.089 MB | 1.77 MB |

All files per prediction with every option (`--drop-data --compress-pae --compress-contact-probs --compress-model`): 0.245 MB
(examples) and 16.1 MB (pools), 14.1x and 9.2x smaller than zip.

## Alternatives considered

- Specialised compression of coordinates (e.g. foldcomp, internal coordinates): left for later, as in the spec.
- Deriving columns from the sequence (atom names, residue names, chain ids): almost free with dictionary encoding already.
- Storing the four tables as text compressed by squashfs: 1.9x larger.
