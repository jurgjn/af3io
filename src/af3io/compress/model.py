"""
    Codec for predicted structures (*model.cif): the file is split into sections at # lines (one CIF category each),
    and every section of at least MIN_SIZE bytes is replaced by a reference line to a blob /blobs/model/<category>/<ab>/<sha256>,
    named after its first item (e.g. _atom_site); text before the data_ line (the terms of use) goes to /blobs/model/header/.
    The data_ line and # lines stay in the stub. A line starting with af3io:// is not valid CIF, so references cannot be
    confused with contents:

        data_pools_5k_0040f80
        #
        _entry.id pools_5k_0040f80
        #
        af3io:///blobs/model/_atom_site/3f/3fa9...

    Loops of large tables (PARQUET) are stored as parquet files with typed columns (integers, fixed-point decimals as
    integers, dictionary-encoded strings) and the formatting (column widths, header) in the parquet metadata. Other
    sections, and loops that cannot be re-formatted exactly, are stored as text (compressed by squashfs).
"""
import io, json, re

import numpy as np

from .codec import Span
from .container import blob_path, blob_ref, sha256

PARQUET = ('_atom_site', '_entity_poly_seq', '_ma_qa_metric_local', '_pdbx_poly_seq_scheme')
MIN_SIZE = 256 # smaller sections stay inline
PARAMETERS = { 'min_size': MIN_SIZE, 'parquet': list(PARQUET), 'parquet_zstd_level': 19 }
LOOP_FORMAT_VERSION = 1

_CATEGORY = re.compile(rb'(_[A-Za-z0-9_\-]+)\.')
_REF_LINE = re.compile(rb'^af3io:///(blobs/model/\S+)\n', re.MULTILINE)
_NULLS = (b'.', b'?')

def enabled(settings):
    return settings.model

def _category(section):
    # Name of the category of a section from its first item, e.g. _atom_site for loop_ followed by _atom_site.group_PDB
    for line in section:
        if line.strip() != b'loop_':
            match = _CATEGORY.match(line)
            return match[1].decode() if match is not None else None
    return None

def split(data, settings):
    lines = data.splitlines(keepends=True)
    data_line = next((i for i, line in enumerate(lines) if line.startswith(b'data_')), None)
    if data_line is None:
        raise ValueError('no data_ line')
    stub, spans = [], []
    def add(section, category):
        raw = b''.join(section)
        if category is None or len(raw) < MIN_SIZE:
            stub.append(raw)
            return
        span = Span(blob_path('model', category, sha256(raw)), category, raw, 'model')
        spans.append(span)
        stub.append(blob_ref(span.path).encode() + b'\n')
    add(lines[:data_line], 'header')
    stub.append(lines[data_line])
    section = []
    for line in lines[data_line + 1:]:
        if line.rstrip(b'\r\n') == b'#':
            add(section, _category(section))
            stub.append(line)
            section = []
        else:
            section.append(line)
    add(section, _category(section))
    return b''.join(stub), spans

def join(stub, read):
    return _REF_LINE.sub(lambda match: read(match[1].decode()), stub)

def encode_blob(span, settings):
    if span.field in PARQUET:
        try:
            blob = encode_loop(span.raw)
            if decode_loop(blob) == span.raw:
                return blob
        except ValueError:
            pass
    return span.raw

def decode_blob(field, blob):
    return decode_loop(blob) if blob[:4] == b'PAR1' else blob

# Loops as parquet files: every row of a loop written by AlphaFold 3 is its values, each padded to the width of the
# longest value in its column and followed by a space, e.g. 'ATOM 1    N N   . ALA A 1 1   ? 0.932   -20.839 ... \n'

def _format_int(values, mask, null):
    return [ null if is_null else str(value).encode() for value, is_null in zip(values.tolist(), mask.tolist()) ]

def _format_decimal(values, decimals):
    template = f'%.{decimals}f'.encode()
    return [ template % value for value in (values / 10**decimals).tolist() ]

def _type_column(tokens):
    """(type, array, metadata) of a column of tokens (bytes): int (with an optional null token . or ?), decimal (fixed
    number of decimals, stored as integers), or str; numeric types only if they re-format to exactly the same tokens
    """
    import pyarrow as pa
    array = np.array(tokens)
    nulls = np.isin(array, _NULLS)
    null_tokens = set(array[nulls].tolist())
    if len(null_tokens) <= 1 and not nulls.all():
        null = null_tokens.pop() if null_tokens else b'.'
        try:
            values = np.zeros(len(tokens), dtype=np.int64)
            values[~nulls] = array[~nulls].astype(np.int64)
            if _format_int(values, nulls, null) == tokens:
                return 'int', pa.array(values, mask=nulls), { 'null': null.decode() }
        except (ValueError, OverflowError):
            pass
    if not nulls.any() and b'.' in tokens[0]:
        decimals = len(tokens[0]) - tokens[0].index(b'.') - 1
        try:
            values = np.rint(array.astype(np.float64) * 10**decimals).astype(np.int64)
            if _format_decimal(values, decimals) == tokens:
                return 'decimal', pa.array(values), { 'decimals': decimals }
        except (ValueError, OverflowError):
            pass
    return 'str', pa.array([ token.decode('latin-1') for token in tokens ]), {}

def encode_loop(raw):
    """Loop (text of a CIF section starting with loop_) as a parquet file, raises ValueError if not a padded loop"""
    import pyarrow as pa, pyarrow.compute as pc, pyarrow.parquet as pq
    lines = raw.splitlines(keepends=True)
    if len(lines) < 2 or lines[0] != b'loop_\n':
        raise ValueError('not a loop')
    n_items = next((i for i, line in enumerate(lines[1:]) if not line.startswith(b'_')), len(lines) - 1)
    header = b''.join(lines[:1 + n_items])
    n_rows = len(lines) - 1 - n_items
    tokens = raw[len(header):].split()
    if n_items == 0 or n_rows == 0 or len(tokens) != n_rows * n_items:
        raise ValueError('not a loop of single-word values')
    names = [ line.strip().decode('latin-1').split('.', 1)[-1] for line in lines[1:1 + n_items] ]
    if len(set(names)) != len(names):
        raise ValueError('duplicate items')
    arrays, columns, encodings, dictionary = [], [], {}, []
    for name, i in zip(names, range(n_items)):
        column = tokens[i::n_items]
        type_, array, metadata = _type_column(column)
        if type_ == 'str':
            dictionary.append(name)
        else:
            min_max = pc.min_max(array)
            if -2**31 <= min_max['min'].as_py() and min_max['max'].as_py() < 2**31:
                array = array.cast(pa.int32())
            encodings[name] = 'DELTA_BINARY_PACKED'
        arrays.append(array)
        columns.append({ 'type': type_, 'width': max(map(len, column)), **metadata })
    loop = { 'format': LOOP_FORMAT_VERSION, 'header': header.decode('latin-1'), 'columns': columns }
    table = pa.table(arrays, names=names).replace_schema_metadata({ 'af3io': json.dumps(loop) })
    buf = io.BytesIO()
    pq.write_table(table, buf, compression='zstd', compression_level=PARAMETERS['parquet_zstd_level'], write_statistics=False,
        use_dictionary=dictionary, column_encoding=encodings or None)
    return buf.getvalue()

def decode_loop(blob):
    """Text of a loop from a parquet file written by encode_loop"""
    import pyarrow.parquet as pq
    table = pq.read_table(io.BytesIO(blob))
    loop = json.loads(table.schema.metadata[b'af3io'])
    if loop['format'] != LOOP_FORMAT_VERSION:
        raise ValueError(f'unsupported loop format {loop["format"]}')
    n_rows = table.num_rows
    widths = [ column['width'] for column in loop['columns'] ]
    row_size = sum(widths) + len(widths) + 1
    rows = np.full((n_rows, row_size), ord(' '), dtype=np.uint8)
    rows[:, -1] = ord('\n')
    offset = 0
    for column, array, width in zip(loop['columns'], table.columns, widths):
        if column['type'] == 'int':
            values = array.to_numpy(zero_copy_only=False)
            mask = array.is_null().to_numpy(zero_copy_only=False)
            tokens = _format_int(np.nan_to_num(values).astype(np.int64) if mask.any() else values, mask, column['null'].encode())
        elif column['type'] == 'decimal':
            tokens = _format_decimal(array.to_numpy(), column['decimals'])
        else:
            tokens = [ value.encode('latin-1') for value in array.to_pylist() ]
        if n_rows > 0:
            chars = np.array(tokens, dtype=f'S{width}').view(np.uint8).reshape(n_rows, width)
            rows[:, offset:offset + width] = np.where(chars == 0, ord(' '), chars)
        offset += width + 1
    return loop['header'].encode('latin-1') + rows.tobytes()
