
import collections, contextlib, functools, glob, gzip, itertools, io, json, math, os, re, zipfile
from pprint import pprint
from pathlib import Path

import numpy as np, scipy as sp, pandas as pd

class Predictions:
    """
        Read AlphaFold3 predictions where the output directory from a single job has been compressed as a zip archive
        Handles the change in "file name layout" (introduced in ~b78e215) where every output file now starts with the AlphaFold 3 job name..
    """
    def __init__(self, path):
        # find/assign name (from path)
        self.path = Path(path)
        self.name = self.path.stem
        #print('predictions - path:', self.path)
        #print('predictions - name:', self.name)

        with zipfile.ZipFile(self.path) as fh_zip:
            self.file_list = fh_zip.namelist()

        # Initial file name layout
        if f'{self.name}/ranking_scores.csv' in self.file_list:
            self.ranking_scores_path = f'{self.name}/ranking_scores.csv'
            self._file_layout = 0
        # Changed around b78e215; every file except TERMS_OF_USE.md now starts with the job name
        elif f'{self.name}/{self.name}_ranking_scores.csv' in self.file_list:
            self.ranking_scores_path = f'{self.name}/{self.name}_ranking_scores.csv'
            self._file_layout = 1
        # Fail if this changes again..
        else:
            assert False, 'Cannot find ranking_scores in archive'

        #print(f'Reading ranking_scores from:', self.ranking_scores_path)
        self.ranking_scores = pd.read_csv(self._read(self.ranking_scores_path), sep=',')

        # Paths for top-ranked model/confidences
        self.model_path               = f'{self.name}/{self.name}_model.cif'
        self.summary_confidences_path = f'{self.name}/{self.name}_summary_confidences.json'
        self.confidences_path         = f'{self.name}/{self.name}_confidences.json'

        # Add model/confidence paths as columns to self.ranking_scores
        if self._file_layout == 0:
            self.ranking_scores['model_path'] =               [ *map(lambda seed, sample: f'{self.name}/seed-{seed}_sample-{sample}/model.cif', self.ranking_scores['seed'], self.ranking_scores['sample'])]
            self.ranking_scores['summary_confidences_path'] = [ *map(lambda seed, sample: f'{self.name}/seed-{seed}_sample-{sample}/summary_confidences.json', self.ranking_scores['seed'], self.ranking_scores['sample'])]
            self.ranking_scores['confidences_path'] =         [ *map(lambda seed, sample: f'{self.name}/seed-{seed}_sample-{sample}/confidences.json', self.ranking_scores['seed'], self.ranking_scores['sample'])]
        elif self._file_layout == 1:
            self.ranking_scores['model_path'] =               [ *map(lambda seed, sample: f'{self.name}/seed-{seed}_sample-{sample}/{self.name}_seed-{seed}_sample-{sample}_model.cif', self.ranking_scores['seed'], self.ranking_scores['sample'])]
            self.ranking_scores['summary_confidences_path'] = [ *map(lambda seed, sample: f'{self.name}/seed-{seed}_sample-{sample}/{self.name}_seed-{seed}_sample-{sample}_summary_confidences.json', self.ranking_scores['seed'], self.ranking_scores['sample'])]
            self.ranking_scores['confidences_path'] =         [ *map(lambda seed, sample: f'{self.name}/seed-{seed}_sample-{sample}/{self.name}_seed-{seed}_sample-{sample}_confidences.json', self.ranking_scores['seed'], self.ranking_scores['sample'])]
        else:
            assert False

    @contextlib.contextmanager
    def open(self, file):
        with zipfile.ZipFile(self.path) as fh_zip:
            with fh_zip.open(file) as fh:
                yield fh

    def _read(self, file):
        with zipfile.ZipFile(self.path) as fh_zip:
            with fh_zip.open(file) as fh:
                return io.BytesIO(fh.read())

    def read_summary_confidences(self):
        def parse_(path):
            js = json.load(self._read(path))
            s_ = pd.Series(list(js[col] for col in cols), index=cols)
            return s_

        cols = ['fraction_disordered', 'has_clash', 'iptm', 'ptm', 'ranking_score', 'chain_iptm', 'chain_pair_iptm', 'chain_pair_pae_min', 'chain_ptm']
        summary_confidences_ = pd.DataFrame.from_records(self.ranking_scores['summary_confidences_path'].map(parse_))

        merge_ = pd.concat([
            self.ranking_scores,
            summary_confidences_.drop(['ranking_score'], axis=1), # drop ranking_score as the values in ranking_scores have more significant digits
        ], axis=1)[['seed', 'sample', 'ranking_score', 'fraction_disordered', 'has_clash', 'iptm', 'ptm', 'chain_iptm', 'chain_pair_iptm', 'chain_pair_pae_min', 'chain_ptm', 'model_path', 'summary_confidences_path', 'confidences_path']]
        merge_.insert(loc=0, column='name', value=self.name)
        loc_ = len(merge_.columns) # Insert as last column
        merge_.insert(loc=loc_, column='predictions_path', value=self.path)
        return merge_

def read_summary_confidences(path):
    # Wrapper to "just get the iptm scores"
    p = Predictions(path)
    return p.read_summary_confidences()

def chain_pair_reduce(arr, token_chain_ids, func):
    # Apply a user-defined function to non-overlapping submatrices of a large square matrix (n by n dimension).
    # The user-defined function is applied per submatrix, and returns a scalar.
    # Sub-matrix sizes are defined by a list of submatrix widths; sum of list guaranteed to equal n.
    widths = [sum(1 for _ in v) for k, v in itertools.groupby(token_chain_ids)]
    edges = np.cumsum([0, *widths])
    k = len(widths)
    out = np.empty((k, k))
    for i in range(k):
        for j in range(k):
            chain_pair_block = arr[edges[i]:edges[i+1], edges[j]:edges[j+1]]
            out[i, j] = func(chain_pair_block)
    return out

def gmean_k(arr, k):
    flat = np.asarray(arr).ravel()
    top_k = np.partition(flat, -k)[-k:]
    if np.all(top_k == 0):
        return 0.0
    else:
        return sp.stats.gmean(top_k)

def gmean_k_smallest(arr, k):
    flat = np.asarray(arr).ravel()
    bottom_k = np.partition(flat, k - 1)[:k]
    if np.all(bottom_k == 0):
        return 0.0
    else:
        return sp.stats.gmean(bottom_k)

def _get_metrics(pred, confidences_path):
    with pred.open(confidences_path) as fh:
        js = json.load(fh)

    chain_ids = np.asarray(js['token_chain_ids'])
    contact_probs = np.array(js['contact_probs']) # byte-identical across models (but not seeds)

    # PAE not symmetric, aggregate by taking the more pessimistic value (max)
    pae = np.array(js['pae'])
    pae = np.maximum(pae, pae.T)

    # https://link.springer.com/article/10.1038/s44320-026-00189-7
    # expected_ipTM = -0.036255571 + 0.004470512*sqrt(aa_in_protein1 + aa_in_protein2)
    chain_lengths = np.array([len(list(g)) for k, g in itertools.groupby(chain_ids)])
    chain_pair_lengths = chain_lengths[:, None] + chain_lengths[None, :]
    chain_pair_iptm_expected = -0.036255571 + 0.004470512*np.sqrt(chain_pair_lengths)

    contact_probs_pow2 = np.power(contact_probs, 2)
    contact_probs_pow3 = np.power(contact_probs, 3)
    contact_probs_pow4 = np.power(contact_probs, 4)
    contact_probs_pow8 = np.power(contact_probs, 8)

    scores = collections.OrderedDict([
        ('chain_pair_pae_min_recap', chain_pair_reduce(pae, chain_ids, np.min)),
        ('chain_pair_pae_gmean3',    chain_pair_reduce(pae, chain_ids, lambda arr: gmean_k_smallest(arr, 3))),
        ('chain_pair_pae_gmean5',    chain_pair_reduce(pae, chain_ids, lambda arr: gmean_k_smallest(arr, 5))),
        ('chain_pair_pae_gmean10',   chain_pair_reduce(pae, chain_ids, lambda arr: gmean_k_smallest(arr, 10))),
        ('chain_pair_pae_gmean15',   chain_pair_reduce(pae, chain_ids, lambda arr: gmean_k_smallest(arr, 15))),
        ('chain_pair_contact_probs_max',     chain_pair_reduce(contact_probs, chain_ids, np.max)),
        ('chain_pair_contact_probs_count5',  chain_pair_reduce(contact_probs > .5, chain_ids, np.sum)),
        ('chain_pair_contact_probs_count95', chain_pair_reduce(contact_probs > .95, chain_ids, np.sum)),
        ('chain_pair_contact_probs_sum',     chain_pair_reduce(contact_probs, chain_ids, np.sum)),
        ('chain_pair_contact_probs_pow2',    chain_pair_reduce(contact_probs_pow2, chain_ids, np.sum)),
        ('chain_pair_contact_probs_pow3',    chain_pair_reduce(contact_probs_pow3, chain_ids, np.sum)),
        ('chain_pair_contact_probs_pow4',    chain_pair_reduce(contact_probs_pow4, chain_ids, np.sum)),
        ('chain_pair_contact_probs_pow8',    chain_pair_reduce(contact_probs_pow8, chain_ids, np.sum)),
        ('chain_pair_contact_probs_gmean3',  chain_pair_reduce(contact_probs, chain_ids, lambda arr: gmean_k(arr, 3))),
        ('chain_pair_contact_probs_gmean5',  chain_pair_reduce(contact_probs, chain_ids, lambda arr: gmean_k(arr, 5))),
        ('chain_pair_contact_probs_gmean10', chain_pair_reduce(contact_probs, chain_ids, lambda arr: gmean_k(arr, 10))),
        ('chain_pair_contact_probs_gmean15', chain_pair_reduce(contact_probs, chain_ids, lambda arr: gmean_k(arr, 15))),
        ('chain_pair_iptm_expected', chain_pair_iptm_expected),
    ])
    return scores

def read_summary_scores(path):
    pred = Predictions(path)
    scores = pred.read_summary_confidences()
    custom = pd.DataFrame( [_get_metrics(pred, confidences_path) for confidences_path in scores.confidences_path ] )
    custom['chain_pair_iptm_corrected'] = scores['chain_pair_iptm'] - custom['chain_pair_iptm_expected']
    
    merged = pd.concat([scores, custom], axis=1)
    merged = merged.astype({'predictions_path': str})
    for col_ in custom.columns:
        merged[ col_ ] = merged[ col_ ].apply(np.ndarray.tolist)

    cols = list(scores.columns)
    pos = cols.index('chain_pair_pae_min') + 1
    return merged[ cols[:pos] + list(custom.columns) + cols[pos:] ]

def read_model(path):
    # Wrapper to "just get the top model"
    p = Predictions(path)
    with zipfile.ZipFile(p.path) as fh_zip:
        with fh_zip.open(p.model_path) as fh:
            return fh.read().decode()
