"""
Compress confidences JSONs
- `atom_plddts` ranges from 0 to 100 with two significant digits
- `contact_probs` ranges from 0 to 1, two digits
- `pae` ranges from 0 to 99.9, one digit

Notes:
- [contact_probs seems symmetric, pae somewhat symmetric](https://github.com/google-deepmind/alphafold3/issues/619)
- confidences JSON output uses the standard `json` module [with post-processing](https://github.com/google-deepmind/alphafold3/blob/main/src/alphafold3/model/confidence_types.py)
- `token_chain_ids` and `atom_chain_ids` [don't always use the same set of identifiers](https://github.com/google-deepmind/alphafold3/issues/271)

Similar/relevant: https://ngff.openmicroscopy.org/rfc/9/index.html
"""

import argparse, collections, collections.abc, copy, hashlib, itertools, math, gzip, json, os, os.path, re, string, subprocess, sys

import numpy as np

def increasing_encode(a):
    ranges = []
    for i, j in itertools.pairwise(a):
        if not(i == j - 1):
            ranges.append(f'range(1,{i+1})')
    ranges.append(f'range(1,{j+1})')
    return f'chain({",".join(ranges)})'

def repeating_encode(l):
    return '+'.join([f"['{k}']*{len(list(g))}" for k, g in itertools.groupby(l)])

def symm_encode(arr):
    #return arr * np.tri(*arr.shape, k=0) #https://stackoverflow.com/questions/23839688/how-to-fill-upper-triangle-of-numpy-array-with-zeros-in-place
    return arr[np.triu_indices(arr.shape[0])]

def symm_decode(arr):
    #return np.tril(arr) + np.triu(arr.T, 1)
    def get_dim_(m):
        # n**2 + n -2m = 0
        # m = 6 => 3
        # m = 10 => 4
        return int((-1 + math.sqrt(1+8*m)) / 2)

    arr_triu = np.zeros(shape=(get_dim_(arr.shape[0]), get_dim_(arr.shape[0])))
    ind = np.triu_indices(arr_triu.shape[0])
    arr_triu[ind] = arr
    return arr_triu + np.tril(arr_triu.T, -1)

def psymm_indices(n):
    for i in range(n):
        for j in range(i):
            yield(i, j)
            yield(j, i)
        yield(i, i)

def psymm_encode(arr):
    indices_ = np.array([*psymm_indices(arr.shape[0])])
    return arr[ (indices_[:,0], indices_[:,1]) ]

def psymm_decode(arr):
    dim_ = int(math.sqrt(arr.shape[0]))
    arr_decode = np.zeros(shape=(dim_, dim_))
    for i, (j, k) in enumerate(psymm_indices(dim_)):
        arr_decode[j, k] = arr[i]
    return arr_decode

def decimal_encode(arr, digits, dtype):
    return ((10**digits)*np.array(arr)).round().astype(dtype)

def decimal_decode(arr, digits):
    return np.round(10**(-digits) * np.array(arr), digits).tolist()

def json_encode(js):
    # Preprocessing:
    # - atom_plddts, contact_probs, pae: discretise by converting to numpy uint8/16 to fit min/max
    # - atom_chain_ids, token_chain_ids, token_res_ids: encode as (much shorter) python expressions that can be "restored" with eval()
    return dict(
        atom_chain_ids = repeating_encode(js['atom_chain_ids']),
        atom_plddts = decimal_encode(js['atom_plddts'], 2, np.uint16),
        contact_probs = symm_encode(decimal_encode(js['contact_probs'], 2, np.uint8)),
        pae = decimal_encode(js['pae'], 1, np.uint16),
        token_chain_ids = repeating_encode(js['token_chain_ids']),
        token_res_ids = increasing_encode(js['token_res_ids']),
    )

def read_confidences_json(file):
    with open(file) as fh:
        conf = json.load(fh)
    return conf

def format_json(atom_chain_ids, atom_plddts, contact_probs, pae, token_chain_ids, token_res_ids):
    json_str_ = f'''{{
  "atom_chain_ids": {atom_chain_ids},
  "atom_plddts": {atom_plddts},
  "contact_probs": {contact_probs},
  "pae": {pae},
  "token_chain_ids": {token_chain_ids},
  "token_res_ids": {token_res_ids}
}}'''
    return json_str_

def dumps_json(atom_chain_ids, atom_plddts, contact_probs, pae, token_chain_ids, token_res_ids):
    return format_json(
        atom_chain_ids = json.dumps(atom_chain_ids).replace(' ', ''),
        atom_plddts = json.dumps(atom_plddts).replace(' ', '').replace('NaN', 'null'),
        contact_probs = json.dumps(contact_probs).replace(' ', '').replace('NaN', 'null'),
        pae = json.dumps(pae).replace(' ', '').replace('NaN', 'null'),
        token_chain_ids = json.dumps(token_chain_ids).replace(' ', ''),
        token_res_ids = json.dumps(token_res_ids).replace(' ', ''),
    )

def write_json(file, atom_chain_ids, atom_plddts, contact_probs, pae, token_chain_ids, token_res_ids):
    with open(file, 'w') as fh:
        fh.write(format_json(dumps_json(atom_chain_ids, atom_plddts, contact_probs, pae, token_chain_ids, token_res_ids)))
