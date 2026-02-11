
"""
Compress A3 confidences .json

- extract & store the two large matrices as .png-s (via oxipng)
    - contact_probs ranges from 0 to 1, two digits
    - pae ranges from 0 to 99.9, one digit:
    - both seem? symmetric?

Note confidences json encoding is customised:
    https://github.com/google-deepmind/alphafold3/blob/main/src/alphafold3/model/confidence_types.py

https://github.com/strukturag/libheif/issues/562
"""

import argparse, collections, collections.abc, copy, hashlib, itertools, gzip, json, os, os.path, re, string, subprocess, sys
from pathlib import Path
import numpy as np, zarr

def compress_increasing(a):
    ranges = []
    for i, j in itertools.pairwise(a):
        if not(i == j - 1):
            ranges.append(f'range(1,{i+1})')
    ranges.append(f'range(1,{j+1})')
    return f'chain({",".join(ranges)})'

def compress_repeating(l):
    return '+'.join([f"['{k}']*{len(list(g))}" for k, g in itertools.groupby(l)])

def compress_symm(arr):
    return arr * np.tri(*arr.shape, k=0) #https://stackoverflow.com/questions/23839688/how-to-fill-upper-triangle-of-numpy-array-with-zeros-in-place

def decompress_symm(arr):
    return np.tril(arr) + np.triu(arr.T, 1)

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
    atom_chain_ids_str = json.dumps(atom_chain_ids).replace(' ', '')
    atom_plddts_str = json.dumps(atom_plddts).replace(' ', '').replace('NaN', 'null')
    contact_probs_str = json.dumps(contact_probs).replace(' ', '').replace('NaN', 'null')
    pae_str = json.dumps(pae).replace(' ', '').replace('NaN', 'null')
    token_chain_ids_str = json.dumps(token_chain_ids).replace(' ', '')
    token_res_ids_str = json.dumps(token_res_ids).replace(' ', '')

    json_str_ = f'''{{
  "atom_chain_ids": {atom_chain_ids_str},
  "atom_plddts": {atom_plddts_str},
  "contact_probs": {contact_probs_str},
  "pae": {pae_str},
  "token_chain_ids": {token_chain_ids_str},
  "token_res_ids": {token_res_ids_str}
}}'''
    return json_str_

def write_json(file, atom_chain_ids, atom_plddts, contact_probs, pae, token_chain_ids, token_res_ids):
    atom_chain_ids_str = json.dumps(atom_chain_ids).replace(' ', '')
    atom_plddts_str = json.dumps(atom_plddts).replace(' ', '').replace('NaN', 'null')
    contact_probs_str = json.dumps(contact_probs).replace(' ', '').replace('NaN', 'null')
    pae_str = json.dumps(pae).replace(' ', '').replace('NaN', 'null')
    token_chain_ids_str = json.dumps(token_chain_ids).replace(' ', '')
    token_res_ids_str = json.dumps(token_res_ids).replace(' ', '')

    json_str_ = f'''{{
  "atom_chain_ids": {atom_chain_ids_str},
  "atom_plddts": {atom_plddts_str},
  "contact_probs": {contact_probs_str},
  "pae": {pae_str},
  "token_chain_ids": {token_chain_ids_str},
  "token_res_ids": {token_res_ids_str}
}}'''
    with open(file, 'w') as fh:
        fh.write(json_str_)
