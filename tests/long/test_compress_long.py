import concurrent.futures, hashlib, io, multiprocessing, zipfile, zlib
try:
    from compression import zstd # Python >= 3.14
except ImportError:
    from backports import zstd
from pathlib import Path
import pandas as pd, pytest, af3io.compress
from testdata import decompressed_contents, fetch_test_dir

DATASETS = {
    'alphafold3_examples': 'alphafold3_examples/alphafold3_predictions',
    'mgen_pools_5k_top3': 'mgen_pools_5k_top3',
}

# Compression settings added cumulatively, see design/specs/test_compress_efficency_long.md
STAGES = [
    ('squashfs', dict(data='store', pae='none', contact_probs=False, model=False)),
    ('--compress-data', dict(data='compress')),
    ('--drop-data', dict(data='drop')),
    ('--compress-pae', dict(pae='lossless')),
    ('--compress-contact-probs', dict(contact_probs=True)),
    ('--compress-model', dict(model=True)),
    ('--compress-pae-lossy', dict(pae='lossy')),
]

# Categories of the efficiency plot (left to right), with the container categories they are made of
CATEGORIES = {
    'other': ['confidences', 'other', 'squashfs'],
    'model': ['model'],
    'contact_probs': ['contact_probs'],
    'pae': ['pae'],
    'data': ['data'],
}

def sha256_dir(root):
    root = Path(root)
    return { path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in root.rglob('*') if path.is_file() }

@pytest.mark.parametrize('dataset', DATASETS)
def test_compress_roundtrip_long(dataset, tmp_path):
    source = fetch_test_dir(DATASETS[dataset])
    image = tmp_path / f'{dataset}.af3io.sqfs'
    af3io.compress.compress([source], image)
    af3io.compress.decompress(image, tmp_path / 'out')
    assert sha256_dir(tmp_path / 'out') == { name: hashlib.sha256(data).hexdigest() for name, data in decompressed_contents(source).items() }

def confidences_parts(data):
    # Split a confidences JSON (keys sorted by AlphaFold 3) into PAE, contact probabilities, and the rest
    i, j, k = data.index(b'"contact_probs": '), data.index(b'"pae": '), data.index(b'"token_chain_ids": ')
    return { 'pae': data[j:k], 'contact_probs': data[i:j], 'other': data[:i] + data[k:] }

def zstd_size(data, level=19, block_size=1 << 20):
    # Compressed size in a squashfs image, approximately
    return sum(len(zstd.compress(data[i:i + block_size], level)) for i in range(0, len(data), block_size))

def kind_category(name):
    kind = af3io.compress.classify(name)
    return 'other' if kind == 'confidences' else kind

def uncompressed_sizes(contents):
    sizes = dict.fromkeys(CATEGORIES, 0)
    for name, data in contents.items():
        if af3io.compress.classify(name) == 'confidences':
            for part, part_data in confidences_parts(data).items():
                sizes[part] += len(part_data)
        else:
            sizes[kind_category(name)] += len(data)
    return sizes

def zip_sizes(contents):
    # Every file zipped individually (deflate, level 6 as zip -r); PAE/contact probabilities in confidences JSONs are
    # split in proportion to their sizes when compressed separately
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for name, data in contents.items():
            zf.writestr(name, data)
    sizes = dict.fromkeys(CATEGORIES, 0)
    with zipfile.ZipFile(buf) as zf:
        for info in zf.infolist():
            if af3io.compress.classify(info.filename) == 'confidences':
                parts = { part: len(zlib.compress(part_data, 6)) for part, part_data in confidences_parts(contents[info.filename]).items() }
                for part, size in parts.items():
                    sizes[part] += info.compress_size * size / sum(parts.values())
            else:
                sizes[kind_category(info.filename)] += info.compress_size
    sizes['other'] += len(buf.getvalue()) - sum(sizes.values()) # headers, central directory
    return sizes

def _confidences_zstd_sizes(data):
    return { part: zstd_size(part_data) for part, part_data in confidences_parts(data).items() }

def container_sizes(image, confidences_zstd):
    # Compressed sizes of a container by category; parts of confidences JSONs that are not stored as separate blobs are
    # split in proportion to their sizes compressed separately with zstd (confidences_zstd)
    with af3io.compress.open(image) as reader:
        info = reader.info()['compressed']
    sizes = { category: sum(info[cat] for cat in cats) for category, cats in CATEGORIES.items() }
    inline = [ part for part in ('pae', 'contact_probs') if info[part] == 0 ]
    if len(inline) > 0 and info['confidences'] > 0:
        parts = { part: sum(sizes_[part] for sizes_ in confidences_zstd) for part in [*inline, 'other'] }
        for part in inline:
            share = info['confidences'] * parts[part] / sum(parts.values())
            sizes[part] += share
            sizes['other'] -= share
    return sizes

def test_compress_efficiency(record, tmp_path):
    """Sizes of AlphaFold 3 predictions uncompressed, zipped, and in af3io containers with cumulatively more
    aggressive compression settings, see design/specs/test_compress_efficency_long.md
    """
    rows = []
    def add_rows(dataset, n_predictions, stage, settings, sizes):
        for category, size in sizes.items():
            rows.append({ 'dataset': dataset, 'n_predictions': n_predictions, 'stage': stage, 'settings': settings, 'category': category, 'bytes': round(size) })

    for dataset, prefix in DATASETS.items():
        source = fetch_test_dir(prefix)
        contents = decompressed_contents(source)
        n_predictions = sum(name.endswith('ranking_scores.csv') for name in contents)
        add_rows(dataset, n_predictions, 'uncompressed', '', uncompressed_sizes(contents))
        add_rows(dataset, n_predictions, 'zip', '', zip_sizes(contents))
        with concurrent.futures.ProcessPoolExecutor(8, mp_context=multiprocessing.get_context('forkserver')) as executor:
            unique = { hashlib.sha256(data).digest(): data for name, data in contents.items() if af3io.compress.classify(name) == 'confidences' }
            confidences_zstd = list(executor.map(_confidences_zstd_sizes, unique.values()))

        settings = {}
        for i, (stage, stage_settings) in enumerate(STAGES):
            settings.update(stage_settings)
            image = tmp_path / f'{dataset}_{i}.af3io.sqfs'
            af3io.compress.compress([source], image, af3io.compress.Settings(**settings))
            out = tmp_path / f'{dataset}_{i}'
            af3io.compress.decompress(image, out) # verifies checksums (of lossy contents with lossy compression)
            assert sorted(path.relative_to(out).as_posix() for path in out.rglob('*') if path.is_file()) == \
                sorted(name for name in contents if settings['data'] != 'drop' or not name.endswith('_data.json'))
            add_rows(dataset, n_predictions, stage, ' '.join(f'{key}={value}' for key, value in settings.items()), container_sizes(image, confidences_zstd))
            image.unlink()
        del contents

    df = record(pd.DataFrame(rows))
    totals = df.groupby(['dataset', 'stage'], sort=False)['bytes'].sum().unstack('dataset')
    print(totals)
    for dataset in DATASETS:
        stages = totals[dataset]
        assert stages['zip'] < stages['uncompressed']
        assert stages.iloc[-1] < stages['zip'] # af3io beats zip
        assert (stages.iloc[2:].diff().dropna() <= 0.01 * stages['zip']).all() # every stage helps (or does not hurt)
