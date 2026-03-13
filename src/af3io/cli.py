
import collections, contextlib, copy, filecmp, glob, io, itertools, json, os, os.path, subprocess, time, zipfile, warnings
import click, humanfriendly, numpy as np, zarr
import af3io
from pathlib import Path
from pprint import pprint

@click.group(help='Eclectic utilities for AlphaFold3')
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
    AlphaFold3 (alphanumeric lower case and -._).
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
    be used as input for AlphaFold3 inference, skipping the data pipeline step.

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
    as input for the AlphaFold3 data pipeline. The data pipeline output can
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

@cli.command(short_help='Extract MSA/templates from data JSON to separate files')
@click.argument('json_path', type=click.Path(exists=True, file_okay=True, readable=True, path_type=Path))
def data_dump(json_path):
    """Extract MSA/templates from data JSON to separate files.
    Output files will be named according to the input file:
    {json_path}_{id}_{pairedMsa}.a3m
    {json_path}_{id}_{unpairedMsa}.a3m
    """
    js = af3io.input.read(json_path)
    for seq_type, seq_fields in af3io.input.iter_sequences(js):

        for field in ['unpairedMsa', 'pairedMsa']:
            path_a3m = json_path.with_name(f"{json_path.stem}_{seq_fields['id']}_{field}.a3m")
            with open(path_a3m, 'w') as fh:
                fh.write(seq_fields[field])
            click.echo(f"{path_a3m}: {humanfriendly.format_size(len(seq_fields[field]))}")

        for index, template in enumerate(seq_fields['templates']):
            name_cif = template['mmcif'].split()[0].removeprefix('data_')
            path_cif = json_path.with_name(f"{json_path.stem}_{seq_fields['id']}_templates_{index}_{name_cif}.cif")
            with open(path_cif, 'w') as fh:
                fh.write(template['mmcif'])
            click.echo(f"{path_cif}:\twrote {humanfriendly.format_size(len(template['mmcif']))}")
            click.echo(f"queryIndices[:5]:\t{template['queryIndices'][:5]}")
            click.echo(f"templateIndices[:5]:\t{template['templateIndices'][:5]}")

@cli.command(short_help='Compress a confidences JSON')
@click.argument('confidences_json', type=click.Path(exists=True, file_okay=True, readable=True, path_type=Path))
def confidences_compress(confidences_json):
    """Compress a confidences JSON."""

    # Compress from <name>.json to <name>.json.af3io
    confidences_compressed = confidences_json.with_suffix(confidences_json.suffix + '.af3io')

    js = af3io.confidences.read_confidences_json(confidences_json)
    js_encode = af3io.confidences.json_encode(js)

    # Store as a single compressed file using zarr ZipStore
    with zarr.storage.ZipStore(confidences_compressed, mode='w') as store:
        root = zarr.create_group(store=store)

        with warnings.catch_warnings(): # Suppress seemingly harmless `UserWarning: Duplicate name: 'zarr.json'`
            warnings.simplefilter('ignore')
            # atom_plddts, contact_probs, pae: store as attributes
            for name in ['atom_chain_ids', 'token_chain_ids', 'token_res_ids']:
                root.attrs[name] = js_encode[name]

        # atom_plddts, contact_probs, pae: compress with Blosc (PCodec?)
        for name in ['atom_plddts', 'contact_probs', 'pae']:
            array_ = root.create_array(name=name,
                shape=js_encode[name].shape,
                dtype=js_encode[name].dtype,
                compressors=zarr.codecs.BloscCodec(clevel=9),
            )
            if len(js_encode[name].shape) == 1:
                array_[:] = js_encode[name]
            else:
                array_[:,:] = js_encode[name]

    #pools_5k_0040f80/pools_5k_0040f80_confidences.json :  4.45%   (234068369 => 10407605 bytes, pools_5k_0040f80/pools_5k_0040f80_confidences.json.zst)
    source_bytes = os.path.getsize(confidences_json)
    target_bytes = os.path.getsize(confidences_compressed)
    frac_compressed = 100 * target_bytes / source_bytes
    click.echo(f'{confidences_json} : {frac_compressed:.2f}% ({source_bytes} => {target_bytes} bytes, {confidences_compressed})')

@cli.command(short_help='Decompress a confidences JSON')
@click.argument('compressed_json', type=click.Path(exists=True, file_okay=True, readable=True, path_type=Path))
def confidences_decompress(compressed_json):
    """Decompress a confidences JSON."""

    with zarr.storage.ZipStore(compressed_json, read_only=True) as store:
        root = zarr.open_group(store=store, mode='r')
        confidences_str_ = af3io.confidences.dumps_json(
            atom_chain_ids = af3io.confidences_eval.repeating_decode(root.attrs['atom_chain_ids']),
            atom_plddts = af3io.confidences.decimal_decode(root['atom_plddts'], 2),
            contact_probs = af3io.confidences.decimal_decode(af3io.confidences.symm_decode(root['contact_probs']), 2),
            pae = af3io.confidences.decimal_decode(root['pae'], 1),
            token_chain_ids = af3io.confidences_eval.repeating_decode(root.attrs['token_chain_ids']),
            token_res_ids = af3io.confidences_eval.increasing_decode(root.attrs['token_res_ids']),
        )

    # By default, decompress from <name>.json.af3io to <name>.json
    suffixes_ = compressed_json.suffixes
    assert suffixes_[-1] == '.af3io'
    assert suffixes_[-2] == '.json'
    confidences_decompressed = compressed_json.with_suffix('')

    # If <name>.json exists, decompress to <name>.json.decompressed instead
    confidences_decompressed_alt = compressed_json.with_suffix('.decompressed')
    if confidences_decompressed.is_file():
        click.echo(f'{confidences_decompressed} exists, writing to {confidences_decompressed_alt}')
        confidences_decompressed = confidences_decompressed_alt

    # Write decompressed contents
    with open(confidences_decompressed, 'w') as fh:
        fh.write(confidences_str_)
    source_bytes = os.path.getsize(compressed_json)
    click.echo(f'{compressed_json} : {source_bytes} bytes decompressed to {confidences_decompressed}')

@cli.command(short_help='Show compact summary of a confidences JSON')
@click.argument('confidences_json', type=click.Path(exists=True, file_okay=True, readable=True, path_type=Path))
def confidences_show(confidences_json):
    """Show a compact summary of a confidences JSON."""

    js = af3io.confidences.read_confidences_json(confidences_json)

    def array_str_(arr):
        array_ = np.array(arr)
        min_ = array_.min()
        max_ = array_.max()
        nunique_ = len(np.unique(arr))

        if len(array_.shape) == 2:
            no_symmetric = (array_ == array_.T).sum()
            no_elements = array_.shape[0] * array_.shape[1]
            symmetric_ = f' / {no_symmetric:,} of {no_elements:,} symmetric ({100*no_symmetric / no_elements:.1f}%)'
        else:
            symmetric_ = ''

        return f'<shape: {array_.shape} / min: {min_} / max: {max_} / nunique: {nunique_:,}{symmetric_}>'

    click.echo(af3io.confidences.format_json(
        atom_chain_ids = af3io.confidences.repeating_encode(js['atom_chain_ids']),
        atom_plddts = array_str_(js['atom_plddts']),
        contact_probs = array_str_(js['contact_probs']),
        pae = array_str_(js['pae']),
        token_chain_ids = af3io.confidences.repeating_encode(js['token_chain_ids']),
        token_res_ids = af3io.confidences.increasing_encode(js['token_res_ids']),
    ))
