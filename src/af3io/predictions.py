
import functools, glob, gzip, itertools, io, json, math, os, re, zipfile
from pprint import pprint
from pathlib import Path

import pandas as pd

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
        ], axis=1)[['seed', 'sample', 'ranking_score', 'fraction_disordered', 'has_clash', 'iptm', 'ptm', 'chain_iptm', 'chain_pair_iptm', 'chain_pair_pae_min', 'chain_ptm']]
        merge_.insert(loc=0, column='name', value=self.name)
        return merge_

def read_summary_confidences(path):
    # Wrapper to "just get the iptm scores"
    p = Predictions(path)
    return p.read_summary_confidences()
