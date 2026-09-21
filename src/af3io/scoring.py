"""
References:
- actifpTM paper: https://doi.org/10.1093/bioinformatics/btaf107
- ipSAE preprint: https://www.biorxiv.org/content/10.1101/2025.02.10.637595v2
- Original ColabFold actifpTM: https://github.com/sokrypton/ColabFold/blob/main/colabfold/alphafold/extra_ptm.py
- Kuhlman-Lab AlphaFold3 implementation: https://github.com/Kuhlman-Lab/alphafold3
- LIS/LIA/iLIS (Local Interaction Score): https://github.com/flyark/AFM-LIS
- pDockQ: https://doi.org/10.1038/s41467-022-28865-w
- pDockQ2: https://doi.org/10.1093/bioinformatics/btad424
- pDockQ/pDockQ2 cutoffs/reference implementation: https://github.com/DunbrackLab/IPSAE/blob/main/ipsae.py
- Pinc: https://doi.org/10.1002/pro.70760, reference implementation: https://git.mpi-cbg.de/tothpetroczylab/Pinc
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

def mean_symm(arr, decimals=3):
    # Symmetrise via the arithmetic mean of both directions, as used for LIS/cLIS in https://github.com/flyark/AFM-LIS
    return np.round((arr + arr.T) / 2, decimals=decimals)

def sum_symm(arr, decimals=0):
    # Symmetrise via the sum of both directions, as used for LIA/cLIA in https://github.com/flyark/AFM-LIS
    return np.round(arr + arr.T, decimals=decimals)

def _pae_to_ptm(pae_block, d0):
    # https://github.com/google-deepmind/alphafold3/blob/v3.0.4/src/alphafold3/model/network/confidence_head.py#L303-L306
    # substitute with pae, eq (7) of the ipSAE preprint
    return 1.0 / (1 + np.square(pae_block) / np.square(d0))

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

    tm_adjusted_pae = _pae_to_ptm(pae_block, d0)

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

# https://github.com/flyark/AFM-LIS defines the confident interface as PAE <= 12 (A), unchanged for AlphaFold3
LIS_PAE_CUTOFF = 12

def _lis_transform(pae_block, pae_cutoff):
    # https://github.com/flyark/AFM-LIS: 1 - PAE/cutoff for PAE < cutoff, else 0
    transformed = np.zeros_like(pae_block, dtype=float)
    mask = pae_block < pae_cutoff
    transformed[mask] = 1.0 - pae_block[mask] / pae_cutoff
    return transformed

def lis(pae_block, pae_cutoff=LIS_PAE_CUTOFF):
    # Local Interaction Score: https://github.com/flyark/AFM-LIS
    transformed = _lis_transform(pae_block, pae_cutoff)
    mask = transformed > 0
    return transformed.sum() / (1e-8 + mask.sum())

def lia(pae_block, pae_cutoff=LIS_PAE_CUTOFF):
    # Local Interaction Area: count of token pairs with PAE < cutoff
    # https://github.com/flyark/AFM-LIS
    return np.sum(pae_block < pae_cutoff)

def ilis(contacts_block, pae_block, pae_cutoff=LIS_PAE_CUTOFF):
    # Integrated LIS: geometric mean of LIS and cLIS (LIS restricted to structural contacts_block)
    # https://github.com/flyark/AFM-LIS
    transformed = _lis_transform(pae_block, pae_cutoff)
    mask = (transformed > 0) & contacts_block.astype(bool)
    clis = transformed[mask].sum() / (1e-8 + mask.sum())
    return np.sqrt(lis(pae_block, pae_cutoff) * clis)

def _interface_mean_plddt(contacts_block, plddt_row_block, plddt_col_block):
    # Mean pLDDT over the union of chain_i/chain_j residues involved in >=1 contact
    # https://github.com/DunbrackLab/IPSAE/blob/main/ipsae.py
    plddt_i = plddt_row_block[:, 0][contacts_block.any(axis=1)]
    plddt_j = plddt_col_block[0, :][contacts_block.any(axis=0)]
    return np.concatenate([plddt_i, plddt_j]).mean()

def pdockq(contacts_block, plddt_row_block, plddt_col_block):
    # pDockQ: https://doi.org/10.1038/s41467-022-28865-w, via https://github.com/DunbrackLab/IPSAE/blob/main/ipsae.py
    contacts_block = contacts_block.astype(bool)
    npairs = contacts_block.sum()
    if npairs == 0:
        return 0.0
    mean_plddt = _interface_mean_plddt(contacts_block, plddt_row_block, plddt_col_block)
    x = mean_plddt * np.log10(npairs)
    return 0.724 / (1 + np.exp(-0.052 * (x - 152.611))) + 0.018

def pdockq2(contacts_block, pae_block, plddt_row_block, plddt_col_block):
    # pDockQ2: https://doi.org/10.1093/bioinformatics/btad424, via https://github.com/DunbrackLab/IPSAE/blob/main/ipsae.py
    contacts_block = contacts_block.astype(bool)
    npairs = contacts_block.sum()
    if npairs == 0:
        return 0.0
    mean_plddt = _interface_mean_plddt(contacts_block, plddt_row_block, plddt_col_block)
    mean_ptm = _pae_to_ptm(pae_block, 10.0)[contacts_block].mean()
    x = mean_plddt * mean_ptm
    return 1.31 / (1 + np.exp(-0.075 * (x - 84.733))) + 0.005

def iplddt(contacts_block, plddt_row_block, plddt_col_block):
    # Interface pLDDT: mean pLDDT over interface residues, without pDockQ's sigmoid/log-contact transform
    contacts_block = contacts_block.astype(bool)
    if contacts_block.sum() == 0:
        return 0.0
    return _interface_mean_plddt(contacts_block, plddt_row_block, plddt_col_block)

# https://doi.org/10.1002/pro.70760 fixes the contact radius (both spheres) at 12 (A)
PINC_CONTACT_RADIUS = 12.0

def _sphere_vol(r):
    return (4.0 / 3.0) * np.pi * r**3

def _sphere_intersect_vol(Ru, D, rc):
    # Volume of intersection between an uncertainty sphere (radius Ru, from PAE) and a fixed contact
    # sphere (radius rc), with centres D apart. https://mathworld.wolfram.com/Sphere-SphereIntersection.html
    D = np.maximum(D, 1e-8)
    lens = np.pi * np.square(rc + Ru - D) * (
        np.square(D) + 2 * D * (Ru + rc) - 3 * np.square(Ru - rc)
    ) / (12.0 * D)
    contained = _sphere_vol(np.minimum(rc, Ru))
    vol = np.where(D >= rc + Ru, 0.0, np.where(D <= np.abs(rc - Ru), contained, lens))
    return np.maximum(vol, 0.0)

def _pinc_contact_prob(pae_block, dist_block, contact_radius):
    # https://git.mpi-cbg.de/tothpetroczylab/Pinc: intersection(uncertainty sphere, contact sphere) / uncertainty sphere volume
    pae_safe = np.maximum(pae_block, 1e-8)
    p = _sphere_intersect_vol(pae_safe, dist_block, contact_radius) / _sphere_vol(pae_safe)
    return np.clip(np.where(pae_block > 0, p, 0.0), 0.0, 1.0)

def pinc(pae_block, dist_block, contact_radius=PINC_CONTACT_RADIUS):
    # Pinc: https://doi.org/10.1002/pro.70760, via https://git.mpi-cbg.de/tothpetroczylab/Pinc
    # dist_block: mass-weighted centre-of-mass distances between residues (see get_com_dist in predictions.py)
    mask = dist_block < contact_radius
    if mask.sum() == 0:
        return 0.0
    return _pinc_contact_prob(pae_block, dist_block, contact_radius)[mask].mean()
