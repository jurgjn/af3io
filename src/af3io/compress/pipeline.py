"""
    Compress files into, and decompress files from af3io containers (see af3io.compress.container)
"""
import collections, concurrent.futures, contextlib, dataclasses, fnmatch, functools, importlib.metadata, itertools, json, multiprocessing, os, posixpath, re

from .. import archive
from . import container
from .container import Container, Writer, blob_path, classify, sha256

@dataclasses.dataclass(frozen=True)
class Settings:
    data: str = 'compress'          # data pipeline JSONs: compress (af3io.compress.data), store (as-is), or drop
    pae: str = 'lossless'           # PAE matrices in confidences JSONs: lossless, lossy (af3io.compress.confidences), or none
    contact_probs: bool = True      # contact probability matrices in confidences JSONs (af3io.compress.confidences)
    model: bool = True              # predicted structures (af3io.compress.model)
    extract_archives: bool = True   # unpack archives and compressed files (.tar, .zip, .gz, .zst) before compression
    level: int = 19                 # zstd compression level of the squashfs image
    block_size: int = 1 << 20       # squashfs block size

    def __post_init__(self):
        if self.data not in ('compress', 'store', 'drop'):
            raise ValueError(f'invalid data setting {self.data!r}')
        if self.pae not in ('lossless', 'lossy', 'none'):
            raise ValueError(f'invalid pae setting {self.pae!r}')

    def manifest(self):
        settings = dataclasses.asdict(self)
        settings['codecs'] = { kind: container.codec(kind).PARAMETERS for kind in container.KINDS
            if container.codec(kind) is not None and container.codec(kind).enabled(self) }
        return settings

@dataclasses.dataclass
class FileResult:
    path: str               # name of the file (original path)
    kind: str               # kind of the file in the container (contents/<kind>/<path>), None if dropped
    stub: bytes             # contents stored under contents/<kind>/<path>
    sha256: str             # checksum of the (decompressed) contents
    size: int               # size of the (decompressed) contents
    blobs: list             # [ (blob path, blob contents) ]
    uncompressed: dict      # uncompressed bytes by size category (see container.CATEGORIES)
    fallback: str = None    # kind of the file if it was stored as-is as its compression could not be verified

def _member_dir(name):
    return posixpath.dirname(name.rstrip('/'))

def _safe_path(path, locator):
    path = posixpath.normpath(path)
    if path.startswith('/') or path == '..' or path.startswith('../') or path in ('', '.'):
        raise ValueError(f'unsafe file name {path!r} from {locator}')
    return path

def iter_inputs(sources, extract_archives=True):
    """Yield (path, locator) for every file under sources: path is its name in the container, locator refers to the
    file on disk or inside archives (see af3io.archive). A source is added under its base name, like cp -r. With
    extract_archives, archives (.tar, .zip) and compressed files (.gz, .zst) are replaced by their contents, e.g.
    dir/pred.zip containing pred/pred_model.cif.gz is added as dir/pred/pred_model.cif
    """
    for source in sources:
        yield from _iter_source(source, extract_archives)

def _iter_source(source, extract_archives):
    source = os.fspath(source)
    base = os.path.dirname(os.path.abspath(source))
    if extract_archives:
        for locator in archive.walk(source):
            disk_path, *members = archive.split(locator)
            rel = os.path.relpath(os.path.abspath(disk_path), base).replace(os.sep, '/')
            if len(members) == 0:
                path = archive.strip_compression(rel)
            else:
                path = posixpath.join(posixpath.dirname(rel), *map(_member_dir, members[:-1]), archive.strip_compression(members[-1]))
            yield _safe_path(path, locator), locator
    elif os.path.isdir(source):
        for root, dirs, files in os.walk(source):
            dirs.sort()
            for file in sorted(files):
                disk_path = os.path.join(root, file)
                yield _safe_path(os.path.relpath(os.path.abspath(disk_path), base).replace(os.sep, '/'), disk_path), disk_path
    else:
        yield _safe_path(os.path.basename(os.path.abspath(source)), source), source

def list_inputs(sources, extract_archives=True, threads=32, progress=None):
    """Like list(iter_inputs(sources, extract_archives)), with sources listed in parallel threads: listing is dominated
    by the latency of reading archive headers, e.g. ~40 ms per zip file on network file systems. progress.update(1) is
    called for every source listed.
    """
    sources = list(sources)
    listed = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, min(threads, len(sources)))) as executor:
        try:
            for items in executor.map(lambda source: list(_iter_source(source, extract_archives)), sources):
                listed += items
                if progress is not None:
                    progress.update(1)
        except BaseException:
            executor.shutdown(cancel_futures=True)
            raise
    return listed

_SAMPLE_DIR = re.compile(r'seed-\d+_sample-\d+')

def _group_key(path):
    # Prediction directory of a file, i.e. the parent directory, or the one above for files in seed-*_sample-*/
    parent = posixpath.dirname(path)
    if _SAMPLE_DIR.fullmatch(posixpath.basename(parent)):
        parent = posixpath.dirname(parent)
    return parent

def group_inputs(inputs, max_files=64):
    """Group inputs by prediction directory, so files from the same prediction (e.g. the top-ranked model and its
    copy in seed-*_sample-*/) are compressed together, raises ValueError on duplicate paths
    """
    seen, groups = {}, collections.defaultdict(list)
    for path, locator in inputs:
        if path in seen:
            raise ValueError(f'duplicate file {path} from {archive.relpath(seen[path])} and {archive.relpath(locator)}')
        seen[path] = locator
        groups[_group_key(path)].append((path, locator))
    return [ group[i:i + max_files] for group in groups.values() for i in range(0, len(group), max_files) ]

class VerificationError(ValueError):
    pass

def _has_refs(codec, data):
    # Whether data contains references to blobs (only possible in files not compressed by af3io)
    def read(path):
        raise KeyError(path)
    try:
        codec.join(data, read)
        return False
    except KeyError:
        return True

def _encode(path, kind, codec, data, settings, blobs_done, known):
    stub, spans = codec.split(data, settings)
    if codec.join(stub, { span.path: span.original for span in spans }.__getitem__) != data:
        raise VerificationError(f'{path}: stub does not re-generate the original')
    raws = { span.path: span.raw for span in spans }
    blobs = {}
    for span in spans:
        if span.path in blobs or span.path in blobs_done or (known is not None and span.path in known):
            continue
        blob = codec.encode_blob(span, settings)
        if codec.decode_blob(span.field, blob) != span.raw:
            raise VerificationError(f'{path}: blob {span.path} does not decompress to its contents')
        blobs[span.path] = blob
    blobs_done.update(blobs) # only once all parts of the file are compressed (as it is stored as-is otherwise)
    contents = codec.join(stub, raws.__getitem__) # differs from data with lossy compression
    uncompressed = collections.Counter({ kind: len(contents) })
    for span in spans:
        uncompressed[kind] -= len(span.raw)
        uncompressed[span.category] += len(span.raw)
    return FileResult(path, kind, stub, sha256(contents), len(contents), list(blobs.items()), dict(uncompressed))

def compress_file(path, data, settings, cache=None, known=None):
    """Compress a single file, return FileResult. Blobs in known (paths of blobs written already) and in the cache
    (shared between calls, e.g. for files of the same prediction) are not encoded again.
    """
    kind = classify(path)
    digest = sha256(data)
    cache = {} if cache is None else cache
    blobs_done = cache.setdefault('blobs', set())
    if kind == 'data' and settings.data == 'drop':
        return FileResult(path, None, b'', digest, len(data), [], {})
    if (kind, digest) in cache: # identical file in the same group, e.g. the top-ranked model and its copy
        return dataclasses.replace(cache[kind, digest], path=path, blobs=[])
    codec = container.codec(kind)
    if codec is not None and codec.enabled(settings):
        try:
            result = _encode(path, kind, codec, data, settings, blobs_done, known)
        except Exception: # incl. unexpected formats, see design/specs/compress.md
            result = FileResult(path, 'other', data, digest, len(data), [], {'other': len(data)}, fallback=kind)
    elif codec is not None and _has_refs(codec, data):
        result = FileResult(path, 'other', data, digest, len(data), [], {'other': len(data)})
    else:
        result = FileResult(path, kind, data, digest, len(data), [], {kind: len(data)})
    cache[kind, digest] = result
    return result

def _read_input(locator, settings):
    if settings.extract_archives:
        with archive.open(locator) as fh:
            return fh.read()
    with open(locator, 'rb') as fh:
        return fh.read()

_known = None

def _init_known(known):
    global _known
    _known = known

def _compress_group(group, settings):
    cache, results = {}, []
    for path, locator in group:
        data = _read_input(locator, settings)
        results.append(compress_file(path, data, settings, cache, _known))
    return results

def _imap(executor, fn, iterable, ahead):
    # Like executor.map with at most `ahead` tasks submitted but not yet consumed (results are large)
    futures = collections.deque()
    for item in iterable:
        futures.append(executor.submit(fn, item))
        if len(futures) >= ahead:
            yield futures.popleft().result()
    while futures:
        yield futures.popleft().result()

def default_jobs():
    return len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count()

# Workers are single-threaded: without these, every worker starts a thread per core for numpy (OpenBLAS) and pyarrow,
# which exceeds limits on the number of processes per user on machines with many cores
_WORKER_ENV = { 'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1' }

@contextlib.contextmanager
def _worker_env():
    saved = { var: os.environ.get(var) for var in _WORKER_ENV }
    for var, value in _WORKER_ENV.items():
        os.environ.setdefault(var, value)
    try:
        yield
    finally:
        for var, value in saved.items():
            if value is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = value

@contextlib.contextmanager
def _executor(jobs, initializer=None, initargs=()):
    # Spawned (not forked) workers, which start with the environment at the time they are started
    with _worker_env(), concurrent.futures.ProcessPoolExecutor(max_workers=jobs, mp_context=multiprocessing.get_context('spawn'),
            initializer=initializer, initargs=initargs) as executor:
        yield executor

class _NoProgress:
    def __init__(self, total, label=None):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        pass

    def update(self, n):
        pass

KIND_ORDER = { kind: i for i, kind in enumerate(container.KINDS) }

def _af3io_version():
    # af3io has no package metadata when run from a source checkout without installing it (e.g. pytest, see pythonpath in pyproject.toml)
    try:
        return importlib.metadata.version('af3io')
    except importlib.metadata.PackageNotFoundError:
        return 'unknown'

def compress(sources, dest, settings=Settings(), jobs=None, progress=_NoProgress):
    """Compress files under sources into a new container at dest, return the manifest. progress(total, label) returns a
    context manager with an update(n) method to report progress (e.g. click.progressbar), first of listing sources
    (label 'Listing'), then of compressing files (label 'Compressing').
    """
    jobs = jobs or default_jobs()
    sources = list(sources)
    with progress(len(sources), 'Listing') as bar:
        groups = group_inputs(list_inputs(sources, settings.extract_archives, progress=bar))
    stats = {
        'files': collections.Counter(),
        'fallbacks': collections.Counter(),
        'dropped': collections.Counter(),
        'uncompressed': collections.Counter(),
        'blobs': 0,
    }
    with Writer(dest, level=settings.level, block_size=settings.block_size, processors=jobs) as writer, \
            progress(sum(map(len, groups)), 'Compressing') as bar, multiprocessing.get_context('spawn').Manager() as manager:
        known = manager.dict() # blobs written, shared with workers so they skip encoding them again
        with _executor(jobs, initializer=_init_known, initargs=(known,)) as executor:
            for results in _imap(executor, functools.partial(_compress_group, settings=settings), groups, ahead=2 * jobs):
                # Files are packed into squashfs fragment blocks in the order they are written: write files of the same
                # kind next to each other (e.g. models of all samples), and blobs (mostly compressed already) last
                group_blobs = []
                for result in sorted(results, key=lambda result: KIND_ORDER.get(result.kind, 0)):
                    if result.kind is None:
                        stats['dropped']['files'] += 1
                        stats['dropped']['bytes'] += result.size
                        continue
                    stats['uncompressed'].update(result.uncompressed)
                    stats['files'][result.kind] += 1
                    xattrs = { container.XATTR_SHA256: result.sha256, container.XATTR_SIZE: str(result.size) }
                    if result.fallback is not None:
                        stats['fallbacks'][result.fallback] += 1
                        xattrs[container.XATTR_FALLBACK] = result.fallback
                    group_blobs += result.blobs
                    writer.add(f'contents/{result.kind}/{result.path}', result.stub, xattrs)
                for path, blob in group_blobs:
                    if path not in writer:
                        writer.add(path, blob)
                        known[path] = True
                        stats['blobs'] += 1
                bar.update(len(results))
        manifest = {
            'format': container.FORMAT_VERSION,
            'af3io_version': _af3io_version(),
            'settings': settings.manifest(),
            'stats': { key: dict(value) if isinstance(value, collections.Counter) else value for key, value in stats.items() },
        }
        writer.add(container.MANIFEST, json.dumps(manifest, indent=2).encode())
    return manifest

def _matches(name, patterns):
    # fnmatch-style patterns (* also matches /), a pattern matching a directory selects everything below it
    return any(fnmatch.fnmatchcase(name, pattern) or name.startswith(pattern.rstrip('/') + '/') for pattern in patterns)

_reader = None

def _init_reader(image):
    global _reader
    _reader = Container(image)

def _decompress_files(names, dest):
    for name in names:
        data = _reader.read(name)
        path = os.path.join(dest, *name.split('/'))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as fh:
            fh.write(data)
    return len(names)

def decompress(image, dest, patterns=(), jobs=None, force=False, progress=_NoProgress):
    """Decompress files matching patterns (default: all files) from the container at image into the directory
    dest, return their names. Every file is verified against its checksum. Raises FileExistsError if a file
    exists already (unless force) and KeyError if a pattern matches no files. progress(total, label) reports progress
    as in compress (label 'Decompressing').
    """
    jobs = jobs or default_jobs()
    image, dest = os.path.abspath(image), os.path.abspath(dest) # workers may run in a different directory
    with Container(image) as reader:
        names = reader.namelist()
    if len(patterns) > 0:
        for pattern in patterns:
            if not any(_matches(name, [pattern]) for name in names):
                raise KeyError(f'{pattern} not found in {image}')
        names = [ name for name in names if _matches(name, patterns) ]
    for name in names:
        _safe_path(name, image)
    if not force:
        for name in names:
            path = os.path.join(dest, *name.split('/'))
            if os.path.lexists(path):
                raise FileExistsError(f'{path} already exists')

    batches = [ list(batch) for _, batch in itertools.groupby(names, key=_group_key) ]
    with progress(len(names), 'Decompressing') as bar:
        if jobs == 1:
            global _reader
            with Container(image) as _reader:
                for batch in batches:
                    bar.update(_decompress_files(batch, dest))
        else:
            with _executor(jobs, initializer=_init_reader, initargs=(image,)) as executor:
                for n in _imap(executor, functools.partial(_decompress_files, dest=dest), batches, ahead=2 * jobs):
                    bar.update(n)
    return names
