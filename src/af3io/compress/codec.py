"""
    Common parts of codecs, which compress files of a kind (data, confidences, model) by replacing parts of them with
    references to blobs. A codec is a module with:

        enabled(settings)               whether to use the codec with these settings (af3io.compress.Settings)
        split(data, settings)           => (stub, [Span]), raises ValueError if data cannot be compressed
        encode_blob(span, settings)     => blob (bytes) stored in the container for span
        decode_blob(field, blob)        => raw contents of the blob (Span.raw)
        join(stub, read)                => contents re-generated from stub, read(blob path) returns raw blob contents
        PARAMETERS                      settings of the codec recorded in the manifest
"""
import dataclasses, re

from .container import blob_ref

@dataclasses.dataclass
class Span:
    path: str               # blob path in the container, e.g. blobs/data/unpairedMsa/ab/abcd...
    field: str              # e.g. unpairedMsa
    raw: bytes              # contents of the blob after decoding, i.e. the exact bytes replaced by the reference
    category: str           # size category (see container.CATEGORIES)
    original: bytes = None  # bytes replaced in the original file if different from raw (lossy compression)
    context: object = None  # passed on to encode_blob, e.g. chain boundaries

    def __post_init__(self):
        if self.original is None:
            self.original = self.raw

def splice(data, replacements):
    """data with [(start, end, replacement)] (sorted, non-overlapping) replaced"""
    parts, last = [], 0
    for start, end, replacement in replacements:
        parts += [ data[last:start], replacement ]
        last = end
    parts.append(data[last:])
    return b''.join(parts)

def json_ref(path):
    # Reference to a blob as a JSON string, so a stub stays valid JSON
    return f'"{blob_ref(path)}"'.encode()

_JSON_REF = re.compile(rb'"af3io:///(blobs/[^"\\\s]+)"')

def json_join(stub, read):
    return _JSON_REF.sub(lambda match: read(match[1].decode()), stub)
