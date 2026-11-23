"""
    Codec for data pipeline output (*_data.json): MSAs (unpairedMsa, pairedMsa) and templates of every chain are
    stored as separate blobs, so data pipeline output of the same sequence is stored once in a container, however
    many predictions it is part of. Blobs contain the exact JSON text of the value (a string with quotes, or an array
    of templates) compressed with zstd with a long window.
"""
try:
    from compression import zstd # Python >= 3.14
except ImportError:
    from backports import zstd

from . import jsonspan
from .codec import Span, json_join as join, json_ref, splice
from .container import blob_path, sha256

FIELDS = ('unpairedMsa', 'pairedMsa', 'templates')
MIN_SIZE = 256 # smaller values stay inline
PARAMETERS = { 'zstd_level': 19, 'zstd_window_log': 27 }
_ZSTD_OPTIONS = {
    zstd.CompressionParameter.compression_level: PARAMETERS['zstd_level'],
    zstd.CompressionParameter.window_log: PARAMETERS['zstd_window_log'],
    zstd.CompressionParameter.enable_long_distance_matching: 1,
}

def enabled(settings):
    return settings.data == 'compress'

def _select(path):
    # sequences[i].<protein|rna|dna|ligand>.<field>
    if len(path) == 0 or (path[0] == 'sequences' and len(path) <= 3):
        return jsonspan.DESCEND
    elif path[0] == 'sequences' and len(path) == 4 and path[3] in FIELDS:
        return jsonspan.SPAN
    return jsonspan.SKIP

def split(data, settings):
    spans, replacements = [], []
    for path, start, end in jsonspan.find(data.decode('latin-1'), _select):
        raw = data[start:end]
        if len(raw) >= MIN_SIZE:
            field = path[-1]
            span = Span(blob_path('data', field, sha256(raw)), field, raw, 'data')
            spans.append(span)
            replacements.append((start, end, json_ref(span.path)))
    return splice(data, replacements), spans

def encode_blob(span, settings):
    return zstd.compress(span.raw, options=_ZSTD_OPTIONS)

def decode_blob(field, blob):
    return zstd.decompress(blob)
