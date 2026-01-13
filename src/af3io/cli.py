
import collections, contextlib, filecmp, glob, io, itertools, json, os, os.path, subprocess, time, zipfile
from pathlib import Path
from pprint import pprint

import click
import af3io

@click.group(help='Utilities for AlphaFold 3 input/output files')
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
        js['sequences'].append(collections.OrderedDict([(type_, collections.OrderedDict([('id', id_),('sequence', sequence_)]))]))

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

'''
@cli.command(help='Compress AlphaFold3 predictions')
@click.argument('path', type=click.Path(exists=True, file_okay=True, readable=True, path_type=Path))
@click.argument('compressed_file', type=click.Path(exists=False, file_okay=False, readable=False, path_type=Path))
@click.option('--drop-data-pipeline', is_flag=True, default=True)
@click.option('--confidences-as-pngs', is_flag=True, default=True)
def predictions_compress(path, compressed_file, drop_data_pipeline, confidences_as_pngs):
    path_prefix = path.resolve().parent # prefix to be stripped from all file names
    #click.echo(f'path_prefix: {path_prefix}')
    with zipfile.ZipFile(compressed_file, mode='x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for filename in path.resolve().rglob('*'):
            arcname = os.path.relpath(filename.resolve(), path_prefix)
            if filename.is_file():
                if filename.name.endswith('_data.json') and drop_data_pipeline:
                    click.echo(f'dropping: {arcname}')
                elif filename.name.endswith('confidences.json') and not filename.name.endswith('summary_confidences.json') and confidences_as_pngs:
                    compressed_json = Path(arcname).with_suffix('.compressed.json')
                    contact_prob_png = Path(arcname).with_suffix('.contact_prob.png')
                    pae_png = Path(arcname).with_suffix('.pae.png')

                    click.echo(f'encoding: {arcname}')
                    t0 = time.perf_counter()
                    confdata = confidences.read_confidences_json(filename)
                    click.echo(f'    time: {time.perf_counter() - t0:.1f} sec')

                    click.echo(f'      as: {compressed_json}')
                    t0 = time.perf_counter()
                    zf.writestr(zinfo_or_arcname=str(compressed_json), data=confidences.dumps_json(
                        confidences.compress_repeating(confdata['atom_chain_ids']),
                        confdata['atom_plddts'],
                        None,
                        None,
                        confidences.compress_repeating(confdata['token_chain_ids']),
                        confidences.compress_increasing(confdata['token_res_ids']),
                    ))
                    click.echo(f'    time: {time.perf_counter() - t0:.1f} sec')

                    click.echo(f'      as: {contact_prob_png}')
                    t0 = time.perf_counter()
                    zf.writestr(
                        zinfo_or_arcname=str(contact_prob_png),
                        data=confidences.dump_contact_probs(confdata['contact_probs']),
                        compress_type=zipfile.ZIP_STORED,
                    )
                    click.echo(f'    time: {time.perf_counter() - t0:.1f} sec')

                    click.echo(f'      as: {pae_png}')
                    t0 = time.perf_counter()
                    zf.writestr(
                        zinfo_or_arcname=str(pae_png),
                        data=confidences.dump_pae(confdata['pae']),
                        compress_type=zipfile.ZIP_STORED,
                    )
                    click.echo(f'    time: {time.perf_counter() - t0:.1f} sec')

                else:
                    click.echo(f'  adding: {arcname}')
                    zf.write(filename=filename, arcname=arcname)

@cli.command(help='Decompress AlphaFold3 predictions')
@click.argument('compressed_file', type=click.Path(exists=True, file_okay=True, readable=True, path_type=Path))
@click.argument('exdir', default=None)
def predictions_decompress(compressed_file, exdir):
    click.echo(f'compressed_file: {compressed_file}')
    click.echo(f'exdir: {exdir}')

    with zipfile.ZipFile(compressed_file) as zf:
        for filename in zf.namelist():
            if filename.endswith('confidences.compressed.json'):
                contact_prob_png = filename.removesuffix('.compressed.json') + '.contact_prob.png'
                pae_png = filename.removesuffix('.compressed.json') + '.pae.png'
                filename_decompress = os.path.join(exdir, filename.removesuffix('.compressed.json') + '.json')

                click.echo(f' decode: {filename} to: {filename_decompress}')
                click.echo(f'   from: {contact_prob_png}')
                click.echo(f'   from: {pae_png}')

                with zf.open(filename) as fh:
                    confdata = json.load(fh, object_pairs_hook=collections.OrderedDict)

                os.makedirs(os.path.dirname(filename_decompress), exist_ok=True)
                confidences.write_json(filename_decompress,
                    confidences_eval.decompress_repeating(confdata['atom_chain_ids']),
                    confdata['atom_plddts'],
                    confidences.load_contact_probs(io.BytesIO(zf.read(contact_prob_png))).tolist(),
                    #confidences.load_pae(io.BytesIO(zf.read(pae_png))).tolist(),
                    confidences.load_pae(np.asarray(bytearray(zf.read(pae_png)), dtype=np.uint8)).tolist(),
                    confidences_eval.decompress_repeating(confdata['token_chain_ids']),
                    confidences_eval.decompress_increasing(confdata['token_res_ids']),
                )

            elif filename.endswith('.contact_prob.png') or filename.endswith('.pae.png'):
                click.echo(f'   skip: {filename}')
            else:
                click.echo(f'extract: {filename}')
                zf.extract(filename, path=exdir)

@cli.command(help='Test compressed predictions against reference')
@click.argument('compressed_file', type=click.Path(exists=True, file_okay=True, readable=True, path_type=Path))
@click.argument('test_path', type=click.Path(exists=True, file_okay=True, readable=True, path_type=Path))
def predictions_test(compressed_file, test_path):
    click.echo(f'compressed_file: {compressed_file}')
    click.echo(f'test_path: {test_path}')

    with zipfile.ZipFile(compressed_file) as zf:
        for filename in zf.namelist():
            if Path(filename).is_file():
                with zf.open(filename) as fh_zf:
                    with open(filename) as fh_ref:
                        read_zf = fh_zf.read().decode()
                        read_ref = fh_ref.read()
                        read_eq = read_zf == read_ref
                        click.echo(f'{read_eq}\t{filename}\t{len(read_zf)}\t{len(read_ref)}')
            elif filename.endswith('confidences.compressed.json'):
                contact_prob_png = filename.removesuffix('.compressed.json') + '.contact_prob.png'
                pae_png = filename.removesuffix('.compressed.json') + '.pae.png'
                filename_ref = os.path.join(test_path, filename.removesuffix('.compressed.json') + '.json')
                click.echo('confidences:')
                click.echo(f'{filename}')
                click.echo(f'{contact_prob_png}')
                click.echo(f'{pae_png}')
                click.echo(f'{filename_ref}')

                with zf.open(filename) as fh:
                    confdata_zf = json.load(fh, object_pairs_hook=collections.OrderedDict)

                confdata_ref = confidences.read_confidences_json(filename_ref)
                for k, v in confdata_ref.items():
                    if k in {'atom_chain_ids', 'token_chain_ids',}:
                        click.echo(f'{k}\t{confidences_eval.decompress_repeating(confdata_zf[k])==confdata_ref[k]}')
                    elif k in {'token_res_ids',}:
                        click.echo(f'{k}\t{confidences_eval.decompress_increasing(confdata_zf[k])==confdata_ref[k]}')
                    elif k in {'atom_plddts',}:
                        click.echo(f'{k}\t{confdata_zf[k]==confdata_ref[k]}')

                    elif k == 'contact_probs':
                        contact_probs_zf = confidences.load_contact_probs(io.BytesIO(zf.read(contact_prob_png)))
                        n_equal = (contact_probs_zf == confdata_ref['contact_probs']).sum()
                        click.echo(f'{k}: {n_equal} of {contact_probs_zf.size} elements equal')

                    elif k == 'pae':
                        #pae_zf = confidences.load_pae(io.BytesIO(zf.read(pae_png)))#.tolist()
                        #pae_zf = confidences.load_pae(np.asarray(bytearray(zf.read(pae_png)), dtype=np.uint8))
                        im_ = PIL.Image.open(io.BytesIO(zf.read(pae_png)))
                        print('mode:', im_.mode)
                        pae_zf = np.array(im_) / 256 / 10
                        n_equal = (pae_zf == confdata_ref['pae']).sum()

                        click.echo(f'{k}: {n_equal} of {pae_zf.size} elements equal')
                        #print(pae_zf[0,0], pae_zf[0,1], pae_zf[0,2], pae_zf[0,3], pae_zf[0,4])
                        #print(confdata_ref['pae'][0][0], confdata_ref['pae'][0][1], confdata_ref['pae'][0][2], confdata_ref['pae'][0][3], confdata_ref['pae'][0][4])

                        mapping_ = set([
                            (confdata_ref['pae'][ix][iy], float(pae_zf[ix, iy])) for ix, iy in np.ndindex(pae_zf.shape)
                        ])
                        click.echo(f'{len(mapping_)} unique mappings')
                        for pair_ in sorted(mapping_):
                            click.echo(f'    {pair_}')

                        """
                        print('eq:')
                        count = 10
                        for ix, iy in np.ndindex(pae_zf.shape):
                            a = pae_zf[ix, iy]
                            b = confdata_ref['pae'][ix][iy]
                            if a == b:
                                print(ix, iy, a, b)
                                count -= 1
                            if count <= 0:
                                break

                        print('neq:')
                        count = 1000
                        for ix, iy in np.ndindex(pae_zf.shape):
                            a = pae_zf[ix, iy]
                            b = confdata_ref['pae'][ix][iy]
                            if a != b:
                                print(ix, iy, a, b)
                                count -= 1

                            if count <= 0:
                                exit()
                        """
                    else:
                        click.echo(f'{k} unknown')
                        assert False
'''
