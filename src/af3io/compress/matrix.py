"""
    Matrices in confidences JSONs (pae, contact_probs) as written by AlphaFold 3, e.g. [[0.8,2.8,5.9],[...]]: parse them
    into integers (values x 10^decimals), re-format them exactly, and store them as lossless JPEG XL images, one for
    every pair of chain groups, in an uncompressed zip file:

        matrix.json     {"format": 1, "shape": [n, n], "decimals": 1, "symmetric": false,
                         "chains": [["A", 0, 280], ...], "groups": [{"name": "A", "start": 0, "end": 280}, ...]}
        A_A.jxl         block of rows of group A and columns of group A
        A_B.jxl         ...

    Groups are consecutive chains (runs of token_chain_ids) with at least min_size tokens together, so small chains such
    as ions or ligands are merged with their neighbours (e.g. group B-C-D). For symmetric matrices, only blocks on and
    above the diagonal are stored, with the lower triangle of diagonal blocks set to zero.
"""
import io, itertools, json, re, warnings, zipfile

import numpy as np

FORMAT_VERSION = 1
INDEX = 'matrix.json'
_ZIP_DATE = (1980, 1, 1, 0, 0, 0)
_CHAIN_ID = re.compile(r'[A-Za-z0-9]+')

def parse(raw, decimals):
    """Matrix of non-negative decimal numbers (JSON text as bytes) as integers (values x 10^decimals), raises ValueError
    if raw is not such a matrix. Does not check formatting (compare format(parse(raw)) with raw for that).
    """
    if not (raw.startswith(b'[[') and raw.endswith(b']]')):
        raise ValueError('not a matrix')
    n_rows = raw.count(b'],[') + 1
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            values = np.fromstring(raw[2:-2].replace(b'],[', b','), dtype=np.float64, sep=',')
    except Exception as e:
        raise ValueError(f'not a matrix of numbers: {e}')
    if values.size == 0 or values.size % n_rows != 0:
        raise ValueError('not a matrix')
    values = np.rint(values * 10**decimals)
    if not (values.min() >= 0 and values.max() < 1 << 16):
        raise ValueError('values out of range')
    return values.astype(np.uint16).reshape(n_rows, -1)

def format(values, decimals):
    """Matrix of integers (values x 10^decimals) as JSON text without spaces, with numbers formatted like Python floats
    (as written by AlphaFold 3 with json.dumps), e.g. [[0.8,2.8],[3.0,0.05]]
    """
    n_rows, n_cols = values.shape
    tokens = [ repr(k / 10**decimals).encode() + b',' for k in range(int(values.max()) + 1) ]
    width = max(map(len, tokens))
    chars = np.array(tokens, dtype=f'S{width}')[values].view(np.uint8).reshape(n_rows, n_cols * width)
    keep = chars != 0 # tokens are padded with null bytes
    ends = np.cumsum(keep.sum(axis=1)).tolist()
    flat = chars[keep].tobytes()
    rows = [ flat[start:end - 1] for start, end in zip([0] + ends[:-1], ends) ] # without the trailing comma
    return b'[[' + b'],['.join(rows) + b']]'

def chain_runs(chain_ids):
    """Runs of consecutive tokens of the same chain [(chain, start, end)] from token_chain_ids"""
    runs, start = [], 0
    for chain, run in itertools.groupby(chain_ids):
        end = start + len(list(run))
        runs.append((chain, start, end))
        start = end
    return runs

def chain_groups(runs, min_size):
    """Groups of consecutive runs of chains [(name, start, end)] with at least min_size tokens each (unless there is a
    single group), named after their chains, e.g. B-C-D (or by number if chain ids are not alphanumeric or not unique)
    """
    groups = []
    for chain, start, end in runs:
        if len(groups) > 0 and (groups[-1][2] - groups[-1][1] < min_size or end - start < min_size):
            groups[-1] = (groups[-1][0] + [chain], groups[-1][1], end)
        else:
            groups.append(([chain], start, end))
    names = [ '-'.join(chains) for chains, _, _ in groups ]
    if not all(_CHAIN_ID.fullmatch(chain) for chain, _, _ in runs) or len(set(names)) != len(names):
        names = [ str(i) for i in range(len(groups)) ]
    return [ (name, start, end) for name, (_, start, end) in zip(names, groups) ]

def _jxl_encode(block, effort):
    import imagecodecs
    bits = max(8, int(block.max()).bit_length())
    if bits <= 8:
        return imagecodecs.jpegxl_encode(np.ascontiguousarray(block, dtype=np.uint8), lossless=True, effort=effort)
    return imagecodecs.jpegxl_encode(np.ascontiguousarray(block, dtype=np.uint16), lossless=True, effort=effort, bitspersample=bits)

def _jxl_decode(data):
    import imagecodecs
    return imagecodecs.jpegxl_decode(data)

def _block_name(row, col):
    return f'{row["name"]}_{col["name"]}.jxl'

def encode(values, decimals, chain_ids=None, min_size=64, symmetric=False, effort=7):
    """Store a matrix of integers (see parse) as a zip file of JPEG XL images, one for every pair of chain groups (see
    chain_groups; a single group without chain_ids). With symmetric=True, only blocks on and above the diagonal.
    """
    n = values.shape[0]
    if values.shape != (n, n):
        raise ValueError('matrix is not square')
    runs = chain_runs(chain_ids) if chain_ids is not None and len(chain_ids) == n else [('all', 0, n)]
    groups = chain_groups(runs, min_size)
    index = {
        'format': FORMAT_VERSION,
        'shape': list(values.shape),
        'decimals': decimals,
        'symmetric': symmetric,
        'chains': [ [chain, start, end] for chain, start, end in runs ],
        'groups': [ { 'name': name, 'start': start, 'end': end } for name, start, end in groups ],
    }
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', compression=zipfile.ZIP_STORED) as zf:
        zf.writestr(zipfile.ZipInfo(INDEX, _ZIP_DATE), json.dumps(index))
        for (i, row), (j, col) in _blocks(index):
            block = values[row['start']:row['end'], col['start']:col['end']]
            if symmetric and i == j:
                block = np.triu(block)
            zf.writestr(zipfile.ZipInfo(_block_name(row, col), _ZIP_DATE), _jxl_encode(block, effort))
    return buf.getvalue()

def _blocks(index):
    # Pairs of ((i, row group), (j, column group)) stored
    pairs = itertools.product(enumerate(index['groups']), repeat=2)
    return [ ((i, row), (j, col)) for (i, row), (j, col) in pairs if not (index['symmetric'] and i > j) ]

def read_index(blob):
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        return _read_index(zf)

def _read_index(zf):
    index = json.loads(zf.read(INDEX))
    if index['format'] != FORMAT_VERSION:
        raise ValueError(f'unsupported matrix format {index["format"]}')
    return index

def decode(blob):
    """Matrix of integers (see parse) and the index (matrix.json) of a blob written by encode"""
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        index = _read_index(zf)
        values = np.zeros(index['shape'], dtype=np.uint16)
        for (i, row), (j, col) in _blocks(index):
            values[row['start']:row['end'], col['start']:col['end']] = _jxl_decode(zf.read(_block_name(row, col)))
    if index['symmetric']:
        values = np.triu(values) + np.triu(values, 1).T
    return values, index

def decode_block(blob, chain1, chain2):
    """Block of the matrix (integers, see parse) for rows of chain1 and columns of chain2, decoding a single image"""
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        index = _read_index(zf)
        def locate(chain):
            for _, start, end in filter(lambda run: run[0] == chain, index['chains']):
                for i, group in enumerate(index['groups']):
                    if group['start'] <= start and end <= group['end']:
                        return i, group, start, end
            raise KeyError(f'chain {chain} not found')
        (i, row, row_start, row_end), (j, col, col_start, col_end) = locate(chain1), locate(chain2)
        if index['symmetric'] and i > j:
            block = _jxl_decode(zf.read(_block_name(col, row))).T
        else:
            block = _jxl_decode(zf.read(_block_name(row, col)))
        if index['symmetric'] and i == j:
            block = np.triu(block) + np.triu(block, 1).T
    return block[row_start - row['start']:row_end - row['start'], col_start - col['start']:col_end - col['start']]
