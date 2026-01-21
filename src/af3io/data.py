
import argparse, collections, copy, glob, gzip, itertools, json, os, re, string, sys
from pathlib import Path
from pprint import pprint

import click

from . import input

def create_index(path):
    index_files = list(itertools.chain(
        Path(path).rglob('*_data.json'),
        Path(path).rglob('*_data.json.gz'),
    ))#[:100]

    def item_show_func_(file_path):
        if file_path is not None:
            return file_path.name
        else:
            return ''

    index = dict()
    with click.progressbar(index_files, label='Reading files:', item_show_func=item_show_func_) as index_files_bar:
        for path in index_files_bar:
            for seq_type, seq_fields in input.iter_sequences(input.read(str(path.resolve()))):
                if not(seq_type in index.keys()):
                    index[seq_type] = dict()
                if not(seq_fields['sequence'] in index[seq_type].keys()):
                    index[seq_type][ seq_fields['sequence'] ] = str(path.resolve())
    return index

def lookup(js, index, missing_dir=None):
    """ Match chains by sequence, set dataPath based on the index
    Note: this changes index by adding missing sequences with dataPath set to None
    This is to keep track of missing sequences already identified in earlier input JSON files 
    """
    js = copy.deepcopy(js)
    for seq_type, seq_fields in input.iter_sequences(js):
        try:
            seq_fields['dataPath'] = index[ seq_type ][ seq_fields['sequence'] ]
        except KeyError:
            if (missing_dir is not None):
                # Write monomer JSON for missing sequence
                js_missing = input.init()
                js_missing['name'] = input.sanitised_name(f'{js["name"]}_{seq_fields["id"]}')
                js_missing['sequences'].append(collections.OrderedDict([(seq_type, collections.OrderedDict([('id', seq_fields['id']),('sequence', seq_fields['sequence'])]))]))
                path_missing = os.path.join(missing_dir, f"{js_missing['name']}.json")
                click.echo(f'Write:\t{path_missing}')
                input.write(js_missing, path_missing)

                # Add missing sequence to index
                index[seq_type][ seq_fields['sequence'] ] = None
            else:
                click.echo(f"Sequence not in data index: {seq_type}/{seq_fields['sequence']}")
                raise

    return js

def fill(js):
    js = copy.deepcopy(js)
    for seq_type, seq_fields in input.iter_sequences(js):
        for seq_type_fill, seq_fields_fill in input.iter_sequences(input.read(seq_fields['dataPath'])):
            if seq_type == seq_type_fill and seq_fields['sequence'] == seq_fields_fill['sequence']:
                click.echo(f'\tfill id={seq_fields["id"]} from id={seq_fields_fill["id"]} in {seq_fields["dataPath"]}')
                seq_fields['modifications'] = []
                seq_fields['unpairedMsa'] = seq_fields_fill['unpairedMsa']
                seq_fields['pairedMsa'] = seq_fields_fill['pairedMsa']
                seq_fields['templates'] = seq_fields_fill['templates'].copy()
                del seq_fields['dataPath']

    # Always set by the data pipeline; set here to get identical files
    if not('bondedAtomPairs' in js.keys()):
        js['bondedAtomPairs'] = None
    if not('userCCD' in js.keys()):
        js['userCCD'] = None

    # https://github.com/google-deepmind/alphafold3/blob/v3.0.1/src/alphafold3/common/folding_input.py#L1284-L1290
    default_order = ['dialect', 'version', 'name', 'sequences', 'modelSeeds', 'bondedAtomPairs', 'userCCD']
    for field in default_order:
        js.move_to_end(field)

    return js

'''
def eprint(*args, **kwargs):
    # https://stackoverflow.com/questions/5574702/how-do-i-print-to-stderr-in-python
    print(*args, file=sys.stderr, **kwargs)

def main():
    if sys.stdin.isatty():
        eprint('No input detected, exiting')
        sys.exit(0)

    js = input_json.read(sys.stdin)
    js = fill(js)
    sys.stdout.write(input_json.dumps(js))
'''
