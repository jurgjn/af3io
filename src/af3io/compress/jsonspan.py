"""
    Locate the exact text of values in JSON documents, e.g. to replace them by references to blobs

    Only the objects and arrays on the way to the selected values are walked in Python; everything else is skipped
    with the C scanner of the json module, so large values (MSAs, matrices) are cheap to skip. Selected matrices of
    numbers as written by AlphaFold 3 (e.g. PAE) are located without the scanner, which would build a list of floats.
"""
import json, json.decoder, re

DESCEND, SPAN, SKIP = 'descend', 'span', 'skip'

_WS = re.compile(r'[ \t\n\r]*')
_scan_once = json.JSONDecoder().scan_once
_scanstring = json.decoder.scanstring

def _ws(text, idx):
    return _WS.match(text, idx).end()

def _skip(text, idx):
    try:
        return _scan_once(text, idx)[1]
    except StopIteration:
        raise ValueError(f'invalid JSON value at {idx}')

_MATRIX_CHARS = b'0123456789.,[]'

def _skip_matrix(text, idx):
    # End of a matrix of numbers without spaces at idx, e.g. [[0.8,2.8],[3.0,0.05]], None if there is none. Brackets
    # are only at the start, the end and between rows (],[), but numbers are not checked (left to the caller).
    if not text.startswith('[[', idx):
        return None
    end = text.find(']]', idx)
    if end < 0:
        return None
    end += 2
    n_rows = text.count('],[', idx, end) + 1
    if text.count('[', idx, end) != n_rows + 1 or text.count(']', idx, end) != n_rows + 1:
        return None
    if len(text[idx:end].encode('latin-1', errors='replace').translate(None, _MATRIX_CHARS)) > 0:
        return None
    return end

def find(text, select):
    """Return [(path, start, end)] of values in the JSON text (str) for which select(path) returns SPAN; a path is a
    tuple of object keys and array indices, e.g. ('sequences', 0, 'protein', 'unpairedMsa'). select returns DESCEND
    to look inside an object or array, SKIP otherwise. Raises ValueError if text is not valid JSON.
    To get byte offsets of an ASCII or UTF-8 encoded document, decode it as latin-1 (one character per byte).
    Selected matrices of numbers (e.g. [[0.8,2.8],[3.0,0.05]]) are not checked to contain valid JSON numbers.
    """
    spans = []

    def walk(idx, path):
        idx = _ws(text, idx)
        action = select(path)
        if action == SPAN:
            end = _skip_matrix(text, idx)
            if end is None:
                end = _skip(text, idx)
            spans.append((path, idx, end))
            return end
        char = text[idx:idx + 1]
        if action == DESCEND and char in ('{', '['):
            closing = '}' if char == '{' else ']'
            idx = _ws(text, idx + 1)
            if text[idx:idx + 1] == closing:
                return idx + 1
            index = 0
            while True:
                if char == '{':
                    if text[idx:idx + 1] != '"':
                        raise ValueError(f'expected object key at {idx}')
                    key, idx = _scanstring(text, idx + 1)
                    idx = _ws(text, idx)
                    if text[idx:idx + 1] != ':':
                        raise ValueError(f'expected : at {idx}')
                    idx = walk(idx + 1, (*path, key))
                else:
                    idx = walk(idx, (*path, index))
                    index += 1
                idx = _ws(text, idx)
                if text[idx:idx + 1] == ',':
                    idx = _ws(text, idx + 1)
                elif text[idx:idx + 1] == closing:
                    return idx + 1
                else:
                    raise ValueError(f'expected , or {closing} at {idx}')
        return _skip(text, idx)

    end = _ws(text, walk(0, ()))
    if end != len(text):
        raise ValueError(f'extra data at {end}')
    return spans
