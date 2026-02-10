
import collections, contextlib, copy, filecmp, glob, io, itertools, json, os, os.path, subprocess, time, zipfile
from pathlib import Path
from pprint import pprint

import numpy as np
import click

import af3io

@click.group(help='Eclectic utilities for AlphaFold 3')
@click.version_option(package_name='af3io')
def cli():
    pass

@cli.command(short_help='Show compact summary of an input JSON')
@click.argument('input_json', type=click.Path(exists=True, file_okay=True, readable=True, path_type=Path))
def input_show(input_json):
    """Show a compact summary of an input JSON with data pipeline strings (pairedMsa, unpairedMsa, templates)
    summarised by their size & short hash of the contents.

    Useful for comparing two input JSON files, see input_diff.ipynb.
    """
    af3io.input.pprint(af3io.input.read(str(input_json.resolve())))

@cli.command(short_help='Create an input JSON from sequences as command-line arguments')
@click.option('--version', default=2)
@click.option('--model_seed', default=1)
@click.option('--type', multiple=True)
@click.option('--id', multiple=True)
@click.option('--sequence', multiple=True)
@click.argument('json_path', type=click.Path(file_okay=True, writable=True, path_type=Path))
def input_create(version, model_seed, type, id, sequence, json_path):
    """Create an input JSON from protein/DNA/RNA sequences specified as command-
    line arguments.

    The name field is inferred from json_path and checked for compatibility with
    AlphaFold 3 (alphanumeric lower case and -._).
    """

    # Check name attribute
    name = json_path.stem
    if name == af3io.input.sanitised_name(name):
        click.echo(f'Setting name to: {name}')
    else:
        raise click.UsageError(f'{name}.json is not sanitised (maybe try: {af3io.input.sanitised_name(name)}.json)')

    # Assume protein unless specified otherwise
    if len(type) == 0:
        type = list(itertools.repeat('protein', len(sequence)))

    # Auto-enumerate if --id not specified
    if len(id) == 0:
        id = list(itertools.islice(af3io.input.enumerate_chains(), len(sequence)))

    js = af3io.input.init(
        version = version,
        name = name,
        modelSeeds = [model_seed],
    )
    for type_, id_, sequence_ in zip(type, id, sequence):
        js['sequences'].append(af3io.input.init_sequence(type_, id_, sequence_))

    click.echo(f'Write:\t{str(json_path.resolve())}')
    af3io.input.write(js=js, path=str(json_path.resolve()))

@cli.command(short_help='Copy data pipeline strings from existing output')
@click.option('--write-index', is_flag=True, default=False,
    help='Write a sequence to data JSON lookup table.',
)
@click.option('--data_dir', default=None, multiple=True,
    help='Path to _data.json files, can specify multiple times.',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
)
@click.option('--json_path', default=None, help='Path to input JSON file.',
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
)
@click.option('--input_dir', default=None, help='Path to directory with input JSON files.',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
)
@click.option('--output_dir', default=None, help='Path to output directory.',
    type=click.Path(exists=False, file_okay=False, dir_okay=True, writable=True, path_type=Path),
)
@click.option('--missing_dir', default=None, help='Path for missing sequence JSON files.',
    type=click.Path(exists=False, file_okay=False, dir_okay=True, writable=True, path_type=Path),
)
def data_fill(write_index, data_dir, json_path, input_dir, output_dir, missing_dir):
    """Read an input JSON, fill in data pipeline strings from matching sequences
    found in files under --data_dir. Files produced under --output_dir can then
    be used as input for AlphaFold 3 inference, skipping the data pipeline step.

    Files under --data_dir can be either plain JSON (_data.json) or compressed
    with gzip (_data.json.gz). Can specify --data_dir multiple times.

    Data pipeline output is matched by sequence, there is no need to keep a
    consistent set of sequence identifiers (across projects/people/labs) and/or
    file names while still sharing data pipeline output.

    For more than a handful of files in --data_dir, use --write-index to pre-
    compute a sequence-JSON lookup table. This avoids excessive I/O from
    repeatedly reading every file under --data_dir. The index is stored in
    .af3io_data_index.json under --data_dir as a plain-text JSON.

    If (some) sequences do not have data pipeline output, use --missing_dir to
    generate input JSON files, one per missing sequence. These can then be used
    as input for the AlphaFold 3 data pipeline. The data pipeline output can
    then be added as an additional --data_dir argument.
    """

    if (json_path is not None) and (input_dir is None):
        input_jsons = [ json_path ]
    elif (json_path is None) and (input_dir is not None):
        input_jsons = glob.glob(os.path.join(input_dir, '*.json'))
    elif (json_path is None) and (input_dir is None):
        input_jsons = []
    else:
        assert False, 'Cannot specify both --json_path and --input_dir'

    if (output_dir is not None) and (missing_dir is not None):
        assert False, 'Cannot specify both --output_dir and --missing_dir'

    data_index = { 'protein': dict(), 'dna': dict(), 'rna': dict() }
    for data_i_dir in data_dir:
        data_i_index_path = os.path.join(data_i_dir, '.af3io_data_index.json')
        if os.path.isfile(data_i_index_path):
            click.echo(f'Load data index from: {data_i_index_path}')
            with open(data_i_index_path, 'r') as fh:
                data_i_index = json.load(fh)
        else:
            click.echo(f'Create data index from: {data_i_dir}')
            data_i_index = af3io.data.create_index(data_i_dir)
        click.echo(f'Read {len(data_i_index.get('protein', []))} protein, {len(data_i_index.get('dna', []))} dna, {len(data_i_index.get('rna', []))} rna sequence(s)')

        if write_index:
            click.echo(f'Writing index to: {data_i_index_path}')
            with open(data_i_index_path, 'w') as fh:
                json.dump(data_i_index, fh, indent=2)

        for data_type in ['protein', 'dna', 'rna']:       
            if data_type in data_i_index.keys():
                data_index[data_type].update(data_i_index[data_type])

    click.echo(f'Data index has {len(data_index['protein'])} protein, {len(data_index['dna'])} dna, {len(data_index['rna'])} rna sequence(s)')

    for input_json in input_jsons:
        click.echo(f'Read:\t{input_json}')
        js = af3io.input.read(input_json)
        js = af3io.data.lookup(js, data_index, missing_dir=missing_dir)
        if output_dir is not None:
            js = af3io.data.fill(js)
            output_json = os.path.join(output_dir, Path(input_json).stem.removesuffix('_data') + '_data.json')
            click.echo(f'Write:\t{output_json}')
            af3io.input.write(js, output_json)

@cli.command(short_help='Compress a confidences JSON')
@click.argument('confidences_json', type=click.Path(exists=True, file_okay=True, readable=True, path_type=Path))
def confidences_compress(confidences_json):
    """Compress a confidences JSON."""

    confidences_compressed = confidences_json.with_suffix('.compressed.json')
    confidences_contact_probs_png = confidences_json.with_suffix('.contact_probs.png')
    confidences_pae_png = confidences_json.with_suffix('.pae.png')

    js = af3io.confidences.read_confidences_json(confidences_json)

    click.echo(f'Write: {confidences_compressed}')
    confidences_str_ = af3io.confidences.dumps_json(
        atom_chain_ids=af3io.confidences.compress_repeating(js['atom_chain_ids']),
        atom_plddts=js['atom_plddts'],
        contact_probs=str(confidences_contact_probs_png),
        pae=str(confidences_pae_png),
        token_chain_ids=af3io.confidences.compress_repeating(js['token_chain_ids']),
        token_res_ids=af3io.confidences.compress_increasing(js['token_res_ids']),
    )
    with open(confidences_compressed, 'w') as fh:
        fh.write(confidences_str_)

    click.echo(f'Write: {confidences_contact_probs_png}')
    af3io.confidences.save_contact_probs(confidences_contact_probs_png, js['contact_probs'])

    click.echo(f'Write: {confidences_pae_png}')
    af3io.confidences.save_pae(confidences_pae_png, js['pae'])

@cli.command(short_help='Decompress a confidences JSON')
@click.argument('compressed_json', type=click.Path(exists=True, file_okay=True, readable=True, path_type=Path))
def confidences_decompress(compressed_json):
    """Decompress a confidences JSON."""

    confidences_compressed = compressed_json
    confidences_contact_probs_png = compressed_json.with_suffix('').with_suffix('.contact_probs.png')
    confidences_pae_png = compressed_json.with_suffix('').with_suffix('.pae.png')

    click.echo(f'Read: {confidences_compressed}')
    js_compressed = af3io.confidences.read_confidences_json(confidences_compressed)

    click.echo(f'Read: {confidences_contact_probs_png}')
    contact_probs = af3io.confidences.load_contact_probs(confidences_contact_probs_png)

    click.echo(f'Read: {confidences_pae_png}')
    pae = af3io.confidences.load_pae(confidences_pae_png)

    confidences_decompressed = compressed_json.with_suffix('').with_suffix('.decompressed.json')
    confidences_str_ = af3io.confidences.dumps_json(
        atom_chain_ids=af3io.confidences_eval.decompress_repeating(js_compressed['atom_chain_ids']),
        atom_plddts=js_compressed['atom_plddts'],
        # Use round() to recapitulate significant digits as in: https://github.com/google-deepmind/alphafold3/blob/main/src/alphafold3/model/confidence_types.py#L36-L41
        contact_probs=np.round(contact_probs, 2).tolist(),
        pae=(np.round(pae, 1).tolist()),
        token_chain_ids=af3io.confidences_eval.decompress_repeating(js_compressed['token_chain_ids']),
        token_res_ids=af3io.confidences_eval.decompress_increasing(js_compressed['token_res_ids']),
    )
    click.echo(f'Write: {confidences_decompressed}')
    with open(confidences_decompressed, 'w') as fh:
        fh.write(confidences_str_)

@cli.command(short_help='Show compact summary of a confidences JSON')
@click.argument('confidences_json', type=click.Path(exists=True, file_okay=True, readable=True, path_type=Path))
def confidences_show(confidences_json):
    """Show a compact summary of a confidences JSON."""

    js = af3io.confidences.read_confidences_json(confidences_json)

    def describe_array(name, arr):
        click.echo(f'{name} has shape {arr.shape}')

        click.echo(f'{arr.min()} min')
        click.echo(f'{arr.max()} max')
        click.echo(f'{len(np.unique(arr)):,} unique values')

        is_symmetric = (arr == arr.T).all()
        no_symmetric = (arr == arr.T).sum()
        no_elements = arr.shape[0] * arr.shape[1]
        click.echo(f'{no_symmetric:,} of {no_elements:,} elements symmetric ({100*no_symmetric / no_elements:.1f}%)')
        click.echo('')

    describe_array('contact_probs', np.array(js['contact_probs']))
    describe_array('pae', np.array(js['pae']))

'''
    js_ref = copy.deepcopy(js)

    atom_plddts_ = np.array(js['atom_plddts'])
    contact_probs_ = np.array(js['contact_probs'])

    af3io.confidences.save_pae(confidences_pae_png, js['pae'])

    # Show summary
    js['atom_chain_ids'] = af3io.confidences.compress_repeating(js['atom_chain_ids'])
    js['atom_plddts'] = str(atom_plddts_.shape)
    js['contact_probs'] = str(contact_probs_.shape)
    js['pae'] = '<pae>'
    #js['pae'] = str(pae_im_)
    js['token_chain_ids'] = af3io.confidences.compress_repeating(js['token_chain_ids'])
    js['token_res_ids'] = af3io.confidences.compress_increasing(js['token_res_ids'])
    click.echo(json.dumps(js, indent=2))

    # Sanity check pae
    click.echo('pae:')
    click.echo(af3io.confidences.load_pae(confidences_pae_png)[0])
    click.echo(np.array(js_ref['pae'])[0])

    # Sanity checks on compression
    if js_ref['atom_chain_ids'] == af3io.confidences_eval.decompress_repeating(js['atom_chain_ids']):
        click.echo('match\tatom_chain_id')
    else:
        click.echo('mismatch\tatom_chain_id')

    if js_ref['token_chain_ids'] == af3io.confidences_eval.decompress_repeating(js['token_chain_ids']):
        click.echo('match\ttoken_chain_id')
    else:
        click.echo('mismatch\ttoken_chain_id')

    if js_ref['token_res_ids'] == af3io.confidences_eval.decompress_increasing(js['token_res_ids']):
        click.echo('match\ttoken_res_ids')
    else:
        click.echo('mismatch\ttoken_res_ids')
'''
