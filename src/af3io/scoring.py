"""
References:
- actifpTM paper: https://doi.org/10.1093/bioinformatics/btaf107
- ipSAE preprint: https://www.biorxiv.org/content/10.1101/2025.02.10.637595v2
- Original ColabFold actifpTM: https://github.com/sokrypton/ColabFold/blob/main/colabfold/alphafold/extra_ptm.py
- Kuhlman-Lab AlphaFold3 implementation: https://github.com/Kuhlman-Lab/alphafold3
"""

import itertools
import numpy as np

def chain_pair_reduce(func, token_chain_ids, *arrs):
    # Apply a user-defined function to non-overlapping submatrices of large square matrices (n by n dimension)
    # The user-defined function is applied per submatrix (one per input matrix), and returns a scalar
    widths = [sum(1 for _ in v) for k, v in itertools.groupby(token_chain_ids)]
    edges = np.cumsum([0, *widths])
    k = len(widths)
    out = np.empty((k, k))
    for i in range(k):
        for j in range(k):
            blocks = [a[edges[i]:edges[i+1], edges[j]:edges[j+1]] for a in arrs]
            out[i, j] = func(*blocks)
    return out

def ptm_symm(arr, decimals=3):
    # Symmetrise a pTM-like score, eq (11) from https://doi.org/10.1101/2025.02.10.637595
    arr_symm = np.maximum(arr, arr.T)
    return np.round(arr_symm, decimals=decimals)

def _weighted_ptm_from_pae(weights_block, pae_block, num_tokens, d0_lower_bound=0):
    # https://github.com/google-deepmind/alphafold3/blob/v3.0.4/src/alphafold3/model/network/confidence_head.py#L290-L297
    clipped_num_res = np.maximum(num_tokens, 19)
    d0 = np.maximum(
        # https://doi.org/10.1101/2025.02.10.637595
        # > In the AlphaFold code, the minimum value is set to 19, since L=18 produces a negative number
        1.24 * (clipped_num_res - 15) ** (1.0 / 3) - 1.8,
        # > use a minimum value of 1 for d0, since Yang and Skolnick did not test the fit for proteins shorter 
        # > than 30 amino acids (d0=1 for L~26.5), and the denominator in  Eq. 14 starts to blow up for values << 1.0,
        # > which may not be realistic or helpful
        d0_lower_bound
    )

    # https://github.com/google-deepmind/alphafold3/blob/v3.0.4/src/alphafold3/model/network/confidence_head.py#L303-L306
    # substitute with pae, eq (7) of the ipSAE preprint
    tm_adjusted_pae = 1.0 / (1 + np.square(pae_block) / np.square(d0))

    # https://github.com/google-deepmind/alphafold3/blob/v3.0.4/src/alphafold3/model/confidences.py#L627-L631
    normed_residue_weights = weights_block / (
        1e-8 + np.sum(weights_block, axis=-1, keepdims=True)
    )
    # https://github.com/Kuhlman-Lab/alphafold3/blob/main/src/alphafold3/model/confidences.py#L640
    per_alignment = np.sum(tm_adjusted_pae * normed_residue_weights, axis=-1)
    return per_alignment.max()

def iptm_from_pae(pae_block):
    weights_block = np.ones_like(pae_block)
    num_tokens = sum(pae_block.shape)
    return _weighted_ptm_from_pae(weights_block, pae_block, num_tokens)

def actifptm_from_pae(weights_block, pae_block):
    num_tokens = sum(pae_block.shape)
    return _weighted_ptm_from_pae(weights_block, pae_block, num_tokens)

def ipsae(pae_block, pae_cutoff):
    weights_block = (pae_block < pae_cutoff).astype(float)
    num_tokens = weights_block.sum(axis=1, keepdims=True)
    return _weighted_ptm_from_pae(weights_block, pae_block, num_tokens, d0_lower_bound=1)

def actifpsae(weights_block, pae_block):
    num_tokens = weights_block.max(axis=1).sum() + weights_block.max(axis=0).sum()
    return _weighted_ptm_from_pae(weights_block, pae_block, num_tokens, d0_lower_bound=1)

def reactifptm(contacts_block, pae_block):
    # https://www.biorxiv.org/content/10.64898/2026.08.24.746624v1
    num_tokens = sum(pae_block.shape)
    return _weighted_ptm_from_pae(contacts_block, pae_block, num_tokens, d0_lower_bound=1)
