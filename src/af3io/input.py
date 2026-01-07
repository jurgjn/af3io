
import argparse, collections, collections.abc, copy, hashlib, itertools, gzip, json, os, os.path, re, string, subprocess, sys
from copy import deepcopy
from pathlib import Path
from pprint import pprint
from typing import Dict, List, Tuple, Any, TypeAlias

import humanfriendly

from .core import *

JSON: TypeAlias = dict[str, "JSON"] | list["JSON"] | str | int | float | bool | None #https://github.com/python/typing/issues/182#issuecomment-1320974824

def _encode_indices_arrays(js):
    #https://github.com/google-deepmind/alphafold3/blob/v3.0.1/src/alphafold3/common/folding_input.py#L1294-L1302
    return re.sub(
        r'("(?:queryIndices|templateIndices)": \[)([\s\n\d,]+)(\],?)',
        lambda mtch: mtch[1] + re.sub(r'\n\s+', ' ', mtch[2].strip()) + mtch[3],
        js,
    )

def _sequence_hash(seq):
    return hashlib.sha1(seq.encode()).hexdigest()

def init(dialect='alphafold3', version=2, name=None, modelSeeds=[4], bondedAtomPairs=None, userCCD=None):
    return collections.OrderedDict([
            ('dialect', dialect),
            ('version', version),
            ('name', name),
            ('sequences', []),
            ('modelSeeds', modelSeeds),
            ('bondedAtomPairs', bondedAtomPairs),
            ('userCCD', userCCD),
    ])

def enumerate_chains():
    for chain_id in map(lambda *args: ''.join(*args), itertools.chain(
        itertools.product(string.ascii_uppercase, repeat=1),
        itertools.product(string.ascii_uppercase, repeat=2),
        itertools.product(string.ascii_uppercase, repeat=3),
    )):
        yield chain_id

def from_sequences(*sequences):
    def _get_seq(id, seq):
        return collections.OrderedDict([('protein', collections.OrderedDict([('id', id),('sequence', seq)]))])

    js = init()
    for sequence, chain_id in zip(sequences, enumerate_chains()):
        js['sequences'].append(_get_seq(chain_id, sequence))

    return js

def read(path):
    """Read json while preserving order of keys from file"""
    with _open_r(path) as fh:
        return json.load(fh, object_pairs_hook=collections.OrderedDict)

def pprint(js, max_size=500):
    """Print (part of) json without long MSA strings; TODO - add minimum for sequence!"""
    def iter_(js):
        if isinstance(js, str) or isinstance(js, int) or isinstance(js, list):
            return js
        for k, v in js.items():
            if k in {'unpairedMsa', 'pairedMsa'} and len(v) > max_size:
                seqs_ = uf(len(v.splitlines())//2)
                size_ = humanfriendly.format_size(len(v))
                hash_ = hashlib.sha1(v.encode()).hexdigest()[:6]
                js[k] = f'<{seqs_} sequences, {size_}, hash: {hash_}>'
            elif k == 'templates':
                count_ = ul(v)
                str_ = json.dumps(v, indent=2)
                size_ = humanfriendly.format_size(len(str_))
                hash_ = hashlib.sha1(str_.encode()).hexdigest()[:6]
                js[k] = f'<{count_} templates, {size_}, hash: {hash_}>'
            elif isinstance(v, collections.abc.Mapping):
                js[k] = iter_(v)
            elif isinstance(v, list):
                for i in range(len(v)):
                    v[i] = iter_(v[i])
        return js
    
    js = copy.deepcopy(js)
    print(json.dumps(iter_(js), indent=2))

def dumps(js):
    return _encode_indices_arrays(json.dumps(js, indent=2))

def write(js, path):
    """Write json aiming to match AF3; if path contains {}, replaces with name from js"""
    # Infer path from name attribute
    if '{}' in path:
        path = path.format(js['name'])

    # Infer name attribute from path
    basename = os.path.basename(path).removesuffix('.json')
    if not(basename.startswith(js['name'])):
        js['name'] = basename
        print(f'Inferring name attribute {js["name"]} from path - {path}')

    if os.path.dirname(path) != '':
        os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as fh:
        js_str = _encode_indices_arrays(json.dumps(js, indent=2))
        fh.write(js_str)

def unpack_sequence(js_sequence):
    return next(iter(js_sequence.keys())), next(iter(js_sequence.values()))

def iter_sequences(js):
    return map(unpack_sequence, js['sequences'])

def sanitised_name(s):
    """AF3 job names are lower case, numeric, -._
    https://github.com/google-deepmind/alphafold3/blob/v3.0.1/src/alphafold3/common/folding_input.py#L857-L861
    """
    def is_allowed(c):
        return c.islower() or c.isnumeric() or c in set('-._')
    return ''.join(filter(is_allowed, s.strip().lower().replace(' ', '_')))

'''
def count_tokens(path):
    """Count tokens
    TODO: proteins only, no nucleic acids, no ligands, no PTMs...
    """
    sequences = read_input_json(path)['sequences']
    n_tokens = 0
    for seq in sequences:
        if 'protein' in seq:
            n_chains = len(seq['protein']['id'])
            seq_len = len(seq['protein']['sequence'])
            n_tokens += n_chains * seq_len
    return n_tokens
'''
