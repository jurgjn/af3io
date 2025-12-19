
"""
Compress A3 confidences .json

- extract & store the two large matrices as .png-s (via oxipng)
    - contact_probs ranges from 0 to 1, two digits
    - pae ranges from 0 to 99.9, one digit:
    - both seem? symmetric?

Note confidences json encoding is customised:
    https://github.com/google-deepmind/alphafold3/blob/main/src/alphafold3/model/confidence_types.py
"""

import argparse, collections, collections.abc, copy, hashlib, itertools, gzip, json, os, os.path, re, string, subprocess, sys
from pathlib import Path
import numpy as np, PIL, PIL.Image, oxipng
import imageio.v3 as iio
import cv2

def compress_increasing(a):
    min_ = min(a)
    max_ = max(a)
    if a == list(range(min_, max_ + 1)):
        return f'range({min_},{max_+1})'
    else:
        return a

def compress_repeating(l):
    return 'chain(' + ','.join([f"repeat('{k}',{len(list(g))})" for k, g in itertools.groupby(l)]) + ')'

def dumppng(arr, bit_depth):
    raw = oxipng.RawImage(
        data=arr.tobytes(),
        width=arr.shape[0],
        height=arr.shape[1],
        color_type=oxipng.ColorType.grayscale(),
        bit_depth=bit_depth
    )
    return raw.create_optimized_png()

def savepng(file, arr, bit_depth):
    png = dumppng(arr, bit_depth)
    with open(file, 'wb') as fh:
        fh.write(png)

def loadpng(file, dtype):
    png = PIL.Image.open(file)
    arr = np.array(png.convert('L'), dtype=dtype)
    return arr

def dump_contact_probs(contact_probs):
    arr = (100 * np.array(contact_probs)).round()
    #arr *= np.tri(*arr.shape, k=1) #https://stackoverflow.com/questions/23839688/how-to-fill-upper-triangle-of-numpy-array-with-zeros-in-place
    return dumppng(arr=arr.astype(dtype=np.uint8), bit_depth=8)

def save_contact_probs(file, contact_probs):
    arr = (100 * np.array(contact_probs)).round()
    #arr *= np.tri(*arr.shape, k=1) #https://stackoverflow.com/questions/23839688/how-to-fill-upper-triangle-of-numpy-array-with-zeros-in-place
    savepng(file=file, arr=arr.astype(dtype=np.uint8), bit_depth=8)

def load_contact_probs(file):
    #return np.asarray(PIL.Image.open(file)) / 100
    arr = loadpng(file, dtype=np.uint8) / 100
    #return np.tril(arr) + np.triu(arr.T, 1)
    return arr
    #return arr + arr.T
    #return np.where(arr, arr, arr.T) #https://stackoverflow.com/questions/58718365/fast-way-to-convert-upper-triangular-matrix-into-symmetric-matrix

def dump_pae(pae):
    arr = (10 * np.array(pae)).round()
    #arr *= np.tri(*arr.shape, k=1) #https://stackoverflow.com/questions/23839688/how-to-fill-upper-triangle-of-numpy-array-with-zeros-in-place
    return dumppng(arr=arr.astype(dtype=np.uint16), bit_depth=16)

def save_pae(file, pae):
    arr = (10 * np.array(pae)).round()
    #arr *= np.tri(*arr.shape, k=1) #https://stackoverflow.com/questions/23839688/how-to-fill-upper-triangle-of-numpy-array-with-zeros-in-place
    savepng(file=file, arr=arr.astype(dtype=np.uint16), bit_depth=16)

def load_pae(file):
    #arr = (np.asarray(PIL.Image.open(file)) / 256) / 10 # divide-by-256 needed with bit_depth=16, unclear why
    arr = cv2.imdecode(file, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE) / 256 / 10
    print(arr.shape)
    #arr = iio.imread(file, 'PNG-FI')#) / 10
    #return np.tril(arr) + np.triu(arr.T, 1)
    return arr
    #return arr + arr.T
    #return np.where(arr, arr, arr.T) #https://stackoverflow.com/questions/58718365/fast-way-to-convert-upper-triangular-matrix-into-symmetric-matrix

def read_json(file):
    with open(file) as fh:
        conf = json.load(fh, object_pairs_hook=collections.OrderedDict)
    return conf

def read_confidences_json(file):
    with open(file) as fh:
        conf = json.load(fh)
    return conf

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

def compress(path):
    path_compressed = Path(path).stem + '_compressed.json'
    path_contact_probs = Path(path).stem + '_contact_probs.png'
    path_pae = Path(path).stem + '_pae.png'

    confidences = read_json(path)
    write_json(path_compressed,
        confidences['atom_chain_ids'],
        confidences['atom_plddts'],
        None,
        None,
        confidences['token_chain_ids'],
        confidences['token_res_ids'],
    )
    save_contact_probs(path_contact_probs, confidences['contact_probs'])
    save_pae(path_pae, confidences['pae'])

def decompress(path, path_decompress):
    path_compressed = path + '_compressed.json'
    path_contact_probs = path + '_contact_probs.png'
    path_pae = path + '_pae.png'
    confidences = read_json(path_compressed)
    write_json(path_decompress,
        confidences['atom_chain_ids'],
        confidences['atom_plddts'],
        load_contact_probs(path_contact_probs).tolist(),
        load_pae(path_pae).tolist(),
        confidences['token_chain_ids'],
        confidences['token_res_ids'],
    )

def main_compress():
    parser = argparse.ArgumentParser(description='Compress AF3 confidences .json')
    parser.add_argument(
        'confidences_json',
        help='File to compress',
    )
    parser.add_argument(
        'output',
        nargs='?',
        default=None
    )

    args = parser.parse_args()
    compress(args.confidences_json)

def main_decompress():
    parser = argparse.ArgumentParser(description='Decompress AF3 confidences .json')

    '''
    parser = argparse.ArgumentParser(
        description="Sample random pools minimising overlap"
    )
    parser.add_argument(
        'input_json', 
        "-p", 
        help="Pools to skip"
    )
    parser.add_argument(
        "--max_pool_size", 
        "-s", 
        help="Maximum size for pool",
        default=5120,
        type=int,
    )
    parser.add_argument(
        "--max_pools", 
        "-n", 
        help="Maximum number of pools to sample",
        type=int,
    )
    args = parser.parse_args()
    eprint('--init_pools', args.init_pools)
    eprint('--max_pool_size', args.max_pool_size)
    eprint('--max_pools', args.max_pools)
    '''
