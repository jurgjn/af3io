"""
    Codec for confidences JSONs (*confidences.json): PAE and contact probability matrices are stored as separate blobs of
    lossless JPEG XL images (see af3io.compress.matrix), with one image for every pair of chain groups for fast access to
    chain pairs. Of symmetric matrices (contact probabilities), only blocks on and above the diagonal are stored. Contact
    probabilities are the same for all samples of a seed, so they are stored once per seed. Matrices that cannot be
    re-formatted exactly (e.g. after changes in AlphaFold 3) stay in the JSON.
"""
import json

import numpy as np

from . import jsonspan, matrix
from .codec import Span, json_join as join, json_ref, splice
from .container import REF_SCHEME, ChecksumError, blob_path, sha256

MATRICES = ('pae', 'contact_probs')
DECIMALS = { 'contact_probs': 2, 'pae': 1 }
PARAMETERS = { 'jxl_effort': 7, 'min_group_size': 64 }

def enabled(settings):
    return len(fields(settings)) > 0

def fields(settings):
    return [ field for field in DECIMALS if (field == 'pae' and settings.pae != 'none') or (field == 'contact_probs' and settings.contact_probs) ]

def split(data, settings):
    selected = fields(settings)
    def select(path):
        if len(path) == 0:
            return jsonspan.DESCEND
        elif len(path) == 1 and (path[0] in selected or path[0] == 'token_chain_ids'):
            return jsonspan.SPAN
        return jsonspan.SKIP
    found = { path[0]: (start, end) for path, start, end in jsonspan.find(data.decode('latin-1'), select) }
    chain_ids = json.loads(data[slice(*found['token_chain_ids'])]) if 'token_chain_ids' in found else None

    spans, replacements = [], []
    for field in selected:
        if field not in found:
            continue
        start, end = found[field]
        raw = data[start:end]
        try:
            values = matrix.parse(raw, DECIMALS[field])
            if matrix.format(values, DECIMALS[field]) != raw:
                raise ValueError('formatting differs')
        except ValueError:
            continue # keep as-is
        span = Span(blob_path('confidences', field, sha256(raw)), field, raw, field, context=(values, chain_ids))
        spans.append(span)
        replacements.append((start, end, json_ref(span.path)))
    return splice(data, sorted(replacements)), spans

def encode_blob(span, settings):
    values, chain_ids = span.context
    symmetric = values.shape[0] == values.shape[1] and bool((values == values.T).all())
    return matrix.encode(values, DECIMALS[span.field], chain_ids, PARAMETERS['min_group_size'], symmetric=symmetric, effort=PARAMETERS['jxl_effort'])

def decode_blob(field, blob):
    values, index = matrix.decode(blob)
    return matrix.format(values, index['decimals'])

def read(stub, read_blob, checksum=None):
    """Confidences JSON as a dict with matrices (PAE, contact probabilities) as numpy arrays (float64, as from json.load),
    read_blob(path) returns the (encoded) contents of a blob. Decodes blobs into arrays without parsing them as text; with
    checksum (sha256 of the file), re-generates the file to verify it (raises ChecksumError).
    """
    js = json.loads(stub)
    texts = {}
    for field in MATRICES:
        value = js.get(field)
        if isinstance(value, str) and value.startswith(REF_SCHEME):
            path = value.removeprefix(REF_SCHEME).strip('/')
            values, index = matrix.decode(read_blob(path))
            js[field] = values / 10**index['decimals']
            if checksum is not None:
                texts[path] = matrix.format(values, index['decimals'])
        elif value is not None:
            js[field] = np.array(value, dtype=float)
    if checksum is not None and sha256(join(stub, texts.__getitem__)) != checksum:
        raise ChecksumError('checksum mismatch')
    return js
