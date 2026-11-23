
import collections, contextlib, copy, filecmp, glob, io, itertools, json, os, os.path, subprocess, sys, time, zipfile, warnings
import click, humanfriendly, numpy as np
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

@cli.command(short_help='Set name attribute of input JSONs from file name')
@click.option('--check', is_flag=True, default=False,
    help='Only report mismatches, exit with status 1 if any found.',
)
@click.argument('json_paths', nargs=-1, required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True, path_type=Path))
def fixname(check, json_paths):
    """Set the name attribute of input JSON(s) to match the file name.

    The expected name is the file name without .json and without a trailing
    _data (e.g. foo_data.json => foo). Files with a name that is not sanitised
    for AlphaFold3 (alphanumeric lower case and -._) are reported and left
    unchanged. Gzip-compressed files (.gz) are skipped.

    Use --check to report mismatches without modifying any files.
    """
    n_checked, n_mismatched, n_unsanitised, n_skipped = 0, 0, 0, 0
    for json_path in json_paths:
        if json_path.suffix == '.gz':
            click.echo(f'skip (gzip not supported): {json_path}')
            n_skipped += 1
            continue

        n_checked += 1
        name = json_path.name.removesuffix('.json').removesuffix('_data')
        if name != af3io.input.sanitised_name(name):
            click.echo(f'unsanitised: {json_path} (maybe try: {af3io.input.sanitised_name(name)})')
            n_unsanitised += 1
            continue

        js = af3io.input.read(str(json_path))
        if js.get('name') == name:
            click.echo(f'ok: {json_path}')
            continue

        n_mismatched += 1
        click.echo(f'{json_path}: {js.get('name')!r} -> {name!r}')
        if not check:
            js['name'] = name
            af3io.input.write(js=js, path=str(json_path))

    click.echo(f'{n_checked} checked, {n_mismatched} {'mismatched' if check else 'updated'}, {n_unsanitised} unsanitised, {n_skipped} skipped')
    if check and (n_mismatched + n_unsanitised) > 0:
        raise SystemExit(1)

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

@cli.command(short_help='Collect summary scores from predictions')
@click.option('--jobs', '-j', default=None, type=int, help='Number of parallel worker processes  [default: available CPUs].')
@click.argument('predictions', nargs=-1, required=True, type=click.Path(exists=True, file_okay=True, dir_okay=True, path_type=Path))
@click.argument('output_parquet', type=click.Path(file_okay=True, dir_okay=False, writable=True, allow_dash=True, path_type=Path))
def summary_confidences(jobs, predictions, output_parquet):
    """Read summary confidences and custom interface scores (ipSAE, LIS,
    pDockQ, ...) from predictions, and write them as a single table to
    OUTPUT_PARQUET.

    A prediction is the AlphaFold 3 output directory of a single job, either
    as-is or compressed as a zip file. Individual files in it can be compressed
    (gzip, zstd), e.g. <name>/<name>_confidences.json.gz.

    Each of PREDICTIONS can be a single prediction, a directory, or an archive
    of predictions. Directories and archives are searched recursively, including
    nested archives (tar, zip) with optional compression (gzip, zstd), e.g.
    pools_5k.tar containing one zip file per prediction. Predictions inside
    archives are listed in the predictions_path column as
    pools_5k.tar::pools_5k_0040f80.zip (see af3io.archive). Paths in the
    predictions_path column are relative to the current directory.

    If OUTPUT_PARQUET is -, write the table to stdout as tab-separated text
    instead. Progress messages are written to stderr.

    Prefer uncompressed tar files as compressed tar files (.tar.gz, .tar.zst)
    have to be re-read from the start for every prediction.
    """
    import concurrent.futures, multiprocessing, pandas as pd
    time_start = time.monotonic()
    if jobs is None:
        jobs = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count()
    click.echo(f'Using {jobs} worker processes ({os.cpu_count()} CPUs on this machine)', err=True)
    prediction_paths = [ prediction_path for path in predictions for prediction_path in af3io.predictions.find_predictions(path.absolute()) ]
    click.echo(f'{len(prediction_paths)} predictions found', err=True)
    if len(prediction_paths) == 0:
        raise click.UsageError(f'No predictions found in {", ".join(map(str, predictions))}')

    with concurrent.futures.ProcessPoolExecutor(max_workers=jobs, mp_context=multiprocessing.get_context('forkserver')) as executor:
        futures = [ executor.submit(af3io.predictions.read_summary_scores, path) for path in prediction_paths ]
        with click.progressbar(concurrent.futures.as_completed(futures), length=len(futures), label='Reading predictions', show_pos=True, file=sys.stderr) as bar:
            for _ in bar:
                pass
        summary_confidences = pd.concat([ future.result() for future in futures ], axis=0).reset_index(drop=True)
    summary_confidences['predictions_path'] = summary_confidences['predictions_path'].map(af3io.archive.relpath)
    if str(output_parquet) == '-':
        click.echo(summary_confidences.to_csv(sep='\t', index=False), nl=False)
    else:
        summary_confidences.to_parquet(output_parquet, compression='zstd')
        click.echo(f'Write:\t{output_parquet} ({len(summary_confidences)} rows)', err=True)
    click.echo(f'Finished in {humanfriendly.format_timespan(time.monotonic() - time_start)}', err=True)

def _progressbar(label):
    # Progress bar on stderr for af3io.compress.compress/decompress, called with the total number of files
    return lambda total: click.progressbar(length=total, label=label, show_pos=True, file=sys.stderr)

@cli.command(short_help='Compress AlphaFold 3 inputs/outputs into a single file')
@click.option('--compress-data', 'data', flag_value='compress', default=True,
    help='Compress data pipeline JSONs (*_data.json) by storing MSAs and templates of each sequence once  [default].')
@click.option('--no-compress-data', 'data', flag_value='store', help='Store data pipeline JSONs as-is (compressed by squashfs only).')
@click.option('--drop-data', 'data', flag_value='drop', help='Leave out data pipeline JSONs (e.g. if easy to re-create).')
@click.option('--compress-pae', 'pae', flag_value='lossless', default=True,
    help='Compress PAE matrices in confidences JSONs as images (lossless)  [default].')
@click.option('--compress-pae-lossy', 'pae', flag_value='lossy',
    help='Compress PAE matrices as images, keeping values up to 15 Å and putting larger values into 2 Å bins (lossy).')
@click.option('--no-compress-pae', 'pae', flag_value='none', help='Store PAE matrices as-is (compressed by squashfs only).')
@click.option('--compress-contact-probs/--no-compress-contact-probs', default=True, show_default=True,
    help='Compress contact probabilities in confidences JSONs as images (lossless), once per seed.')
@click.option('--compress-model/--no-compress-model', default=True, show_default=True,
    help='Compress predicted structures (*model.cif), with large tables (e.g. _atom_site) as parquet files.')
@click.option('--lossy', is_flag=True, default=False,
    help='Same as --compress-data --compress-pae-lossy --compress-contact-probs --compress-model.')
@click.option('--extract-archives/--no-extract-archives', default=True, show_default=True,
    help='Unpack archives and compressed files (.tar, .zip, .gz, .zst) before compression.')
@click.option('--files-from', type=click.File('r'), default=None,
    help='File with paths to compress, one per line (in addition to SOURCES).')
@click.option('--jobs', '-j', default=None, type=int, help='Number of parallel worker processes  [default: available CPUs].')
@click.argument('sources', nargs=-1, type=click.Path(exists=True, path_type=Path))
@click.argument('dest', type=click.Path(dir_okay=False, path_type=Path))
def compress(data, pae, compress_contact_probs, compress_model, lossy, extract_archives, files_from, jobs, sources, dest):
    """Compress files and directories in SOURCES into a single file DEST
    (.af3io, a squashfs image), e.g.:

    \b
        af3io compress af3io-testdata/mgen_pools_5k_top3 mgen_pools_5k_top3.af3io

    Each of SOURCES is added under its base name (like cp -r), so the example
    above stores files as mgen_pools_5k_top3/pools_5k_0040f80/... in DEST. By
    default, archives (.tar, .zip) and compressed files (.gz, .zst) are
    replaced by their contents, e.g. confidences.json.zst is stored (and
    decompressed) as confidences.json.

    Data pipeline JSONs (*_data.json) are compressed by storing the MSAs
    and templates of each sequence once, however many predictions use them.
    PAE and contact probability matrices in confidences JSONs are stored as
    lossless images (JPEG XL), contact probabilities once per seed. Large
    tables of predicted structures (*model.cif, e.g. _atom_site) are stored
    as parquet files with typed columns.

    Decompressing reproduces every file bit-identically, which is verified
    for every file at compression and decompression time. With
    --compress-pae-lossy (or --lossy), PAE values above 15 Å are put into 2 Å
    bins (e.g. 15.1 to 17.0 become 16.0, above 29.0 become 31.0), keeping the
    formatting of the files otherwise.
    """
    import af3io.compress
    time_start = time.monotonic()
    if lossy:
        ctx = click.get_current_context()
        for name, value in [('data', 'compress'), ('pae', 'lossless'), ('compress_contact_probs', True), ('compress_model', True)]:
            if ctx.get_parameter_source(name) != click.core.ParameterSource.DEFAULT and ctx.params[name] != value:
                raise click.UsageError(f'--lossy conflicts with the option given for {name.removeprefix("compress_").replace("_", "-")}')
        data, pae, compress_contact_probs, compress_model = 'compress', 'lossy', True, True
    sources = list(sources)
    if files_from is not None:
        for line in files_from:
            if line.strip() != '':
                path = Path(line.rstrip('\n'))
                if not path.exists():
                    raise click.UsageError(f'{path} (from --files-from) does not exist')
                sources.append(path)
    if len(sources) == 0:
        raise click.UsageError('No files to compress, specify SOURCES and/or --files-from')
    if dest.exists():
        raise click.UsageError(f'{dest} already exists')
    settings = af3io.compress.Settings(data=data, pae=pae, contact_probs=compress_contact_probs, model=compress_model, extract_archives=extract_archives)
    try:
        manifest = af3io.compress.compress(sources, dest, settings, jobs=jobs, progress=_progressbar('Compressing'))
    except (ValueError, FileExistsError) as e:
        raise click.ClickException(str(e))
    stats = manifest['stats']
    size_in, size_out = sum(stats['uncompressed'].values()), dest.stat().st_size
    click.echo(f'Write:\t{dest} ({sum(stats["files"].values())} files, {humanfriendly.format_size(size_in)} => {humanfriendly.format_size(size_out)})', err=True)
    if sum(stats['fallbacks'].values()) > 0:
        click.echo(f'Stored as-is as compression could not be verified: {dict(stats["fallbacks"])}', err=True)
    click.echo(f'Finished in {humanfriendly.format_timespan(time.monotonic() - time_start)}', err=True)

@cli.command(short_help='Decompress files from an af3io container')
@click.option('--force', '-f', is_flag=True, default=False, help='Overwrite existing files.')
@click.option('--jobs', '-j', default=None, type=int, help='Number of parallel worker processes  [default: available CPUs].')
@click.argument('image', type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument('patterns', nargs=-1)
@click.argument('dest', type=click.Path(file_okay=False, path_type=Path))
def decompress(force, jobs, image, patterns, dest):
    """Decompress files from IMAGE (an af3io container) into the directory DEST, e.g.:

    \b
        af3io decompress mgen_pools_5k_top3.af3io .

    Optional PATTERNS select files to decompress by name (shell-style
    wildcards, see fnmatch, where * also matches /), a pattern matching a
    directory selects all files under it, e.g.:

    \b
        af3io decompress mgen_pools_5k_top3.af3io 'mgen_pools_5k_top3/pools_5k_0040f80/*' ./new_path

    Every file is verified against the checksum of its original contents.
    """
    import af3io.compress
    try:
        names = af3io.compress.decompress(image, dest, patterns, jobs=jobs, force=force, progress=_progressbar('Decompressing'))
    except (KeyError, FileExistsError, af3io.compress.ChecksumError) as e:
        raise click.ClickException(str(e).strip('\'"'))
    click.echo(f'Write:\t{len(names)} files to {dest}', err=True)
    with af3io.compress.open(image) as reader:
        if reader.manifest['settings'].get('pae') == 'lossy':
            click.echo('Note: PAE matrices were compressed lossily (values above 15 Å in bins, see af3io compress --help)', err=True)

@cli.command(short_help='Show sizes of an af3io container by file type')
@click.argument('image', type=click.Path(exists=True, dir_okay=False, path_type=Path))
def info(image):
    """Show uncompressed and compressed sizes of IMAGE (an af3io container) by
    type of file: data pipeline output (data), predicted structures (model),
    PAE and contact probability matrices (pae, contact_probs), the remaining
    parts of confidences JSONs (confidences), all other files (other), and
    squashfs metadata (squashfs). Does not decompress any files.
    """
    import af3io.compress
    with af3io.compress.open(image) as reader:
        df = reader.info()
        manifest = reader.manifest
    stats = manifest['stats']
    click.echo(f'{image}: af3io {manifest["af3io_version"]}, {sum(stats["files"].values())} files')
    click.echo(f'settings: {", ".join(f"{key}={value}" for key, value in manifest["settings"].items())}')
    if sum(stats['fallbacks'].values()) > 0:
        click.echo(f'fallbacks (stored as-is as compression could not be verified): {stats["fallbacks"]}')
    if stats['dropped'].get('files', 0) > 0:
        click.echo(f'dropped: {stats["dropped"]["files"]} data pipeline JSONs ({humanfriendly.format_size(stats["dropped"]["bytes"])})')
    click.echo(f'{"":15}{"files":>8}{"uncompressed":>14}{"compressed":>12}{"ratio":>8}')
    for cat, row in df.iterrows():
        ratio = f'{row["uncompressed"] / row["compressed"]:.1f}x' if row['compressed'] > 0 and row['uncompressed'] > 0 else ''
        click.echo(f'{cat:15}{row["files"]:8d}{humanfriendly.format_size(row["uncompressed"]):>14}{humanfriendly.format_size(row["compressed"]):>12}{ratio:>8}')
