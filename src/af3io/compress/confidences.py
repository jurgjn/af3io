"""
    Codec for confidences JSONs (*confidences.json): PAE and contact probability matrices are stored as separate blobs of
    lossless JPEG XL images (see af3io.compress.matrix), with one image for every pair of chain groups for fast access to
    chain pairs. Of symmetric matrices (contact probabilities), only blocks on and above the diagonal are stored. Contact
    probabilities are the same for all samples of a seed, so they are stored once per seed. Matrices that cannot be
    re-formatted exactly (e.g. after changes in AlphaFold 3) stay in the JSON.

    With lossy compression of PAE (--compress-pae-lossy), values up to 15 Å are kept, and larger ones are put into bins (see
    PAE_LOSSY_BINS and design/adr/0006-compress-pae-lossy.md); the file is re-generated with the formatting of the
    original and the values of the bins, and its checksum is that of the re-generated file.
"""
import json

import numpy as np

from . import jsonspan, matrix
from .codec import Span, json_join as join, json_ref, splice
from .container import REF_SCHEME, ChecksumError, blob_path, sha256

MATRICES = ('pae', 'contact_probs')
DECIMALS = { 'contact_probs': 2, 'pae': 1 }

# Lossy PAE: (lower, upper, value) in Å, values in (lower, upper] become value (upper None: all larger values), values up to
# the lowest bound stay as they are
PAE_LOSSY_BINS = [ (15, 17, 16), (17, 19, 18), (19, 21, 20), (21, 23, 22), (23, 25, 24), (25, 27, 26), (27, 29, 28), (29, None, 31) ]
PARAMETERS = { 'jxl_effort': 7, 'min_group_size': 64, 'pae_lossy_bins': PAE_LOSSY_BINS }

def quantize_pae(values):
    """PAE matrix of integers (Å x 10, see af3io.compress.matrix.parse) with values in PAE_LOSSY_BINS replaced"""
    lut = np.arange(max(int(values.max()), 10 * PAE_LOSSY_BINS[-1][2]) + 1, dtype=np.uint16)
    for lower, upper, value in PAE_LOSSY_BINS:
        lut[10 * lower + 1:None if upper is None else 10 * upper + 1] = 10 * value
    return lut[values]

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
            json.loads(raw) # matrices are not checked by jsonspan.find, raises ValueError if not valid JSON
            continue # keep as-is
        stored = raw
        if field == 'pae' and settings.pae == 'lossy':
            values = quantize_pae(values)
            stored = matrix.format(values, DECIMALS[field])
        span = Span(blob_path('confidences', field, sha256(stored)), field, stored, field, original=raw, context=(values, chain_ids))
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
