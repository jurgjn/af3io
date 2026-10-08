
import collections, contextlib, functools, glob, gzip, itertools, io, json, math, os, re, zipfile
from pprint import pprint
from pathlib import Path, PurePosixPath

import numpy as np, scipy as sp, pandas as pd
import scipy.special  # not reliably populated on `sp` by `import scipy` alone

import Bio, Bio.PDB

from . import archive
from .scores import chain_pair_reduce, chain_pair_ptm_from_pae, model_confidence, ptm_symm, min_symm, mean_symm, sum_symm, iptm_from_pae, actifptm_from_pae, ipsae, actifpsae, reactifptm, lis, lia, clis, clia, ilis, ilia, n_contacts, n_interface_residues, pdockq, pdockq2, iplddt, pinc

class Predictions:
    """
        Read AlphaFold3 predictions from the output directory of a single job, either as-is or compressed as a zip archive
        Handles the change in "file name layout" (introduced in ~b78e215) where every output file now starts with the AlphaFold 3 job name..
        The zip archive can be nested in other archives and/or compressed, e.g. pools_5k.tar::pools_5k_0040f80.zip (see af3io.archive)
        Individual output files can be compressed (.gz, .zst), e.g. pools_5k_0040f80/pools_5k_0040f80_confidences.json.gz
        File paths (model_path, confidences_path, ...) always refer to uncompressed names relative to the parent of the output directory
    """
    def __init__(self, path):
        # find/assign name (from path); nested locators are kept as strings as they're not valid file system paths
        self.path = str(path) if archive.is_nested(path) else Path(path)
        self._stack = contextlib.ExitStack()
        if isinstance(self.path, Path) and self.path.is_dir():
            self.name = self.path.absolute().name
            self._zip = None
            self.file_list = sorted(f'{self.name}/{Path(root, file).relative_to(self.path).as_posix()}' for root, dirs, files in os.walk(self.path) for file in files)
        else:
            self.name = PurePosixPath(archive.strip_compression(archive.split(path)[-1])).stem
            self._zip = self._stack.enter_context(zipfile.ZipFile(self._stack.enter_context(archive.open(self.path, seekable=True))))
            self.file_list = self._zip.namelist()
        #print('predictions - path:', self.path)
        #print('predictions - name:', self.name)

        # Uncompressed => stored file names, e.g. name/name_model.cif => name/name_model.cif.gz
        self._stored_names = { archive.strip_compression(file): file for file in self.file_list }

        # Initial file name layout
        if f'{self.name}/ranking_scores.csv' in self._stored_names:
            self.ranking_scores_path = f'{self.name}/ranking_scores.csv'
            self._file_layout = 0
        # Changed around b78e215; every file except TERMS_OF_USE.md now starts with the job name
        elif f'{self.name}/{self.name}_ranking_scores.csv' in self._stored_names:
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

    def close(self):
        self._stack.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()

    @contextlib.contextmanager
    def open(self, file):
        stored_name = self._stored_names[file]
        if self._zip is not None:
            fh_stored = self._zip.open(stored_name)
        else:
            fh_stored = open(self.path / PurePosixPath(stored_name).relative_to(self.name), 'rb')
        with fh_stored, archive.decompress(fh_stored, stored_name) as fh:
            yield fh

    def _read(self, file):
        with self.open(file) as fh:
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
    with Predictions(path) as p:
        return p.read_summary_confidences()

def _is_ranking_scores(path):
    # <name>/ranking_scores.csv (initial file name layout) or <name>/<name>_ranking_scores.csv, optionally compressed
    path = PurePosixPath(archive.strip_compression(path))
    return path.name in ('ranking_scores.csv', f'{path.parent.name}_ranking_scores.csv')

def find_predictions(path):
    """Yield paths/locators of predictions under path, which can be a directory, a single prediction, or a
    (nested) archive of predictions, e.g. a tar file of zip files (see af3io.archive). A prediction is either
    a zip file with a top-level <name>/<name>_ranking_scores.csv, or a directory <name> on disk with a
    <name>_ranking_scores.csv (individual files can be compressed, e.g. <name>_ranking_scores.csv.gz).
    """
    seen = set()
    for locator in archive.walk(path):
        *parents, member = archive.split(locator)
        if len(parents) == 0: # file on disk
            prediction = str(Path(member).parent) if _is_ranking_scores(member) and Path(member).parent.name != '' else None
        elif len(PurePosixPath(member).parts) == 2 and _is_ranking_scores(member) \
                and PurePosixPath(archive.strip_compression(parents[-1])).suffix.lower() == '.zip':
            prediction = archive.join(*parents)
        else:
            prediction = None
        if prediction is not None:
            if prediction not in seen:
                seen.add(prediction)
                yield prediction

def pseudo_beta(res):
    if res.get_resname() != 'GLY' and 'CB' in res:
        return res['CB']
    return res['CA'] if 'CA' in res else None

def get_model_coords(struct):
    return np.asarray( [pseudo_beta(res).coord for res in struct.get_residues() ] )

def get_model_bfactors(struct):
    # pLDDT is stored as the B-factor of each atom in AlphaFold3 model.cif files
    return np.asarray( [pseudo_beta(res).get_bfactor() for res in struct.get_residues() ] )

def get_dist(struct):
    coords = get_model_coords(struct)
    dist = sp.spatial.distance.cdist(coords, coords)
    return dist

# https://git.mpi-cbg.de/tothpetroczylab/Pinc: heavy-atom masses (Da) for residue centre-of-mass, unlisted elements default to carbon
ATOMIC_MASS = {'S': 32.0650, 'P': 30.9738, 'O': 15.9994, 'N': 14.0067}

def get_model_com_coords(struct):
    # Mass-weighted centre of mass over heavy atoms per residue, as used by https://git.mpi-cbg.de/tothpetroczylab/Pinc
    def com(res):
        atoms = [a for a in res if a.element != 'H']
        masses = np.array([ATOMIC_MASS.get(a.element, 12.0107) for a in atoms])
        coords = np.array([a.coord for a in atoms])
        return np.average(coords, axis=0, weights=masses)
    return np.asarray( [com(res) for res in struct.get_residues() ] )

def get_com_dist(struct):
    coords = get_model_com_coords(struct)
    dist = sp.spatial.distance.cdist(coords, coords)
    return dist

def read_struct(pred, model_path):
    with pred.open(model_path) as fh_model:
        model_name = Path(model_path).stem
        pdb_str = fh_model.read().decode()
        parser = Bio.PDB.FastMMCIFParser()
        struct = parser.get_structure(model_name, io.StringIO(pdb_str))
        return struct[0]

def _get_metrics(pred, confidences_path, model_path, chain_pair_iptm):
    # Assumes input is protein chains only..
    with pred.open(confidences_path) as fh:
        js = json.load(fh)

    chain_ids = np.asarray(js['token_chain_ids'])
    contact_probs = np.array(js['contact_probs']) # byte-identical across models (but not seeds)

    # PAE not symmetric, different across all models/samples
    pae = np.array(js['pae'])

    # contact matrix, pLDDT (B-factor), and centre-of-mass distances based on model coordinates
    struct = read_struct(pred, model_path)
    isin_8A = get_dist(struct) <= 8
    dist_com = get_com_dist(struct)
    plddt = get_model_bfactors(struct)
    plddt_row = np.broadcast_to(plddt[:, None], pae.shape)
    plddt_col = np.broadcast_to(plddt[None, :], pae.shape)

    # https://link.springer.com/article/10.1038/s44320-026-00189-7
    # expected_ipTM = -0.036255571 + 0.004470512*sqrt(aa_in_protein1 + aa_in_protein2)
    chain_lengths = np.array([len(list(g)) for k, g in itertools.groupby(chain_ids)])
    chain_pair_lengths_sum = chain_lengths[:, None] + chain_lengths[None, :]
    chain_pair_iptm_expected = -0.036255571 + 0.004470512*np.sqrt(chain_pair_lengths_sum)

    chain_pair_iptm_corrected = np.asarray(chain_pair_iptm) - chain_pair_iptm_expected
    chain_pair_ptm = chain_pair_ptm_from_pae(chain_ids, pae)
    chain_pair_ipsae10 = chain_pair_reduce(functools.partial(ipsae, pae_cutoff=10), chain_ids, pae)
    chain_pair_ipsae15 = chain_pair_reduce(functools.partial(ipsae, pae_cutoff=15), chain_ids, pae)
    # LIS family: symmetrise without rounding as iLIS/iLIA are derived from these (as in https://github.com/flyark/AFM-LIS)
    chain_pair_lis = mean_symm(chain_pair_reduce(lis, chain_ids, pae), decimals=None)
    chain_pair_clis = mean_symm(chain_pair_reduce(clis, chain_ids, isin_8A, pae), decimals=None)
    chain_pair_lia = sum_symm(chain_pair_reduce(lia, chain_ids, pae), decimals=None)
    chain_pair_clia = sum_symm(chain_pair_reduce(clia, chain_ids, isin_8A, pae), decimals=None)

    scores = collections.OrderedDict([
        ('chain_pair_iptm_corrected',     np.round(chain_pair_iptm_corrected, 3)),
        ('chain_pair_model_confidence',   np.round(model_confidence(chain_pair_iptm, chain_pair_ptm), 6)),
        ('chain_pair_model_confidence_corrected', np.round(model_confidence(chain_pair_iptm_corrected, chain_pair_ptm), 6)),
        ('chain_pair_iptm_from_pae',      ptm_symm(chain_pair_reduce(iptm_from_pae, chain_ids, pae))),
        ('chain_pair_actifptm_from_pae',  ptm_symm(chain_pair_reduce(actifptm_from_pae, chain_ids, contact_probs, pae))),
        ('chain_pair_ipsae10',            ptm_symm(chain_pair_ipsae10)),
        ('chain_pair_ipsae15',            ptm_symm(chain_pair_ipsae15)),
        ('chain_pair_ipsae10_min',        min_symm(chain_pair_ipsae10)),
        ('chain_pair_ipsae15_min',        min_symm(chain_pair_ipsae15)),
        ('chain_pair_actifpsae',          ptm_symm(chain_pair_reduce(actifpsae, chain_ids, contact_probs, pae))),
        ('chain_pair_reactifptm',         ptm_symm(chain_pair_reduce(reactifptm, chain_ids, isin_8A, pae))),
        ('chain_pair_lis',                np.round(chain_pair_lis, 6)),
        ('chain_pair_lia',                chain_pair_lia),
        ('chain_pair_clis',               np.round(chain_pair_clis, 6)),
        ('chain_pair_clia',               chain_pair_clia),
        ('chain_pair_ilis',               ilis(chain_pair_lis, chain_pair_clis)),
        ('chain_pair_ilia',               ilia(chain_pair_lia, chain_pair_clia)),
        ('chain_pair_pdockq',             ptm_symm(chain_pair_reduce(pdockq, chain_ids, isin_8A, plddt_row, plddt_col))),
        ('chain_pair_pdockq2',            ptm_symm(chain_pair_reduce(pdockq2, chain_ids, isin_8A, pae, plddt_row, plddt_col))),
        ('chain_pair_iplddt',             ptm_symm(chain_pair_reduce(iplddt, chain_ids, isin_8A, plddt_row, plddt_col))),
        ('chain_pair_pinc',               mean_symm(chain_pair_reduce(pinc, chain_ids, pae, dist_com))),
        ('chain_pair_contact_probs_max',  np.round(chain_pair_reduce(np.max, chain_ids, contact_probs), 2)),
        ('chain_pair_contact_probs_pow3', np.round(chain_pair_reduce(lambda contacts_block, distance_mask: np.sum(contacts_block[distance_mask] ** 3), chain_ids, contact_probs, isin_8A), 6)),
        ('chain_pair_contact_probs_pow6', np.round(chain_pair_reduce(lambda contacts_block, distance_mask: np.sum(contacts_block[distance_mask] ** 6), chain_ids, contact_probs, isin_8A), 6)),
        ('chain_pair_contact_probs_pow9', np.round(chain_pair_reduce(lambda contacts_block, distance_mask: np.sum(contacts_block[distance_mask] ** 9), chain_ids, contact_probs, isin_8A), 6)),
        ('chain_pair_n_contacts',         chain_pair_reduce(n_contacts, chain_ids, isin_8A)),
        ('chain_pair_n_interface_residues', chain_pair_reduce(n_interface_residues, chain_ids, isin_8A)),
    ])
    return scores

def read_summary_scores(path):
    with Predictions(path) as pred:
        scores = pred.read_summary_confidences()
        custom = pd.DataFrame( [* map(_get_metrics, itertools.repeat(pred), scores.confidences_path, scores.model_path, scores['chain_pair_iptm']) ] )
    
    merged = pd.concat([scores, custom], axis=1)
    merged = merged.astype({'predictions_path': str})
    for col_ in custom.columns:
        merged[ col_ ] = merged[ col_ ].apply(np.ndarray.tolist)

    cols = list(scores.columns)
    pos = cols.index('chain_pair_iptm') + 1
    return merged[ cols[:pos] + list(custom.columns) + cols[pos:] ]

def read_contact_probs(path, chain1, chain2):
    # contact_probs profile for chain1 residues by aggregating over chain2 residues using aggfunc
    with Predictions(path) as pred:
        scores = pred.read_summary_confidences()
        confidences_path = scores.head(1)['confidences_path'].squeeze()
        with pred.open(confidences_path) as fh:
            js = json.load(fh)

    chain_ids = np.asarray(js['token_chain_ids'])
    contact_probs = np.array(js['contact_probs'])
    block = contact_probs[ np.ix_(chain_ids == chain1, chain_ids == chain2) ]
    return block

def read_chain_contact_probs(path, chain1, chain2, aggfunc=np.max):
    block = read_contact_probs(path, chain1, chain2)
    return aggfunc(block, axis=1)

def read_model(path):
    # Wrapper to "just get the top model"
    with Predictions(path) as p:
        with p.open(p.model_path) as fh:
            return fh.read().decode()
