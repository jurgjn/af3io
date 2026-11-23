"""
Read and summarise confidences JSONs (compression: see af3io.compress.confidences)
- `atom_plddts` ranges from 0 to 100 with two significant digits
- `contact_probs` ranges from 0 to 1, two digits
- `pae` ranges from 0 to 99.9, one digit

Notes:
- [contact_probs seems symmetric, pae somewhat symmetric](https://github.com/google-deepmind/alphafold3/issues/619)
- confidences JSON output uses the standard `json` module [with post-processing](https://github.com/google-deepmind/alphafold3/blob/main/src/alphafold3/model/confidence_types.py)
- `token_chain_ids` and `atom_chain_ids` [don't always use the same set of identifiers](https://github.com/google-deepmind/alphafold3/issues/271)
"""

import itertools, json

def increasing_encode(a):
    ranges = []
    for i, j in itertools.pairwise(a):
        if not(i == j - 1):
            ranges.append(f'range(1,{i+1})')
    ranges.append(f'range(1,{j+1})')
    return f'chain({",".join(ranges)})'

def repeating_encode(l):
    return '+'.join([f"['{k}']*{len(list(g))}" for k, g in itertools.groupby(l)])

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
