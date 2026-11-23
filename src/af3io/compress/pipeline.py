"""
    Compress files into, and decompress files from af3io containers (see af3io.compress.container)
"""
import collections, concurrent.futures, dataclasses, fnmatch, functools, importlib.metadata, itertools, json, multiprocessing, os, posixpath, re

from .. import archive
from . import container
from .container import Container, Writer, blob_path, classify, sha256

@dataclasses.dataclass(frozen=True)
class Settings:
    extract_archives: bool = True   # unpack archives and compressed files (.tar, .zip, .gz, .zst) before compression
    level: int = 19                 # zstd compression level of the squashfs image
    block_size: int = 1 << 20       # squashfs block size

    def manifest(self):
        return dataclasses.asdict(self)

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

def compress_file(path, data, settings, cache=None):
    """Compress a single file, return FileResult"""
    kind = classify(path)
    digest = sha256(data)
    cache = {} if cache is None else cache
    if (kind, digest) in cache: # identical file in the same group, e.g. the top-ranked model and its copy
        return dataclasses.replace(cache[kind, digest], path=path)
    result = FileResult(path, kind, data, digest, len(data), [], {kind: len(data)})
    cache[kind, digest] = result
    return result

def _read_input(locator, settings):
    if settings.extract_archives:
        with archive.open(locator) as fh:
            return fh.read()
    with open(locator, 'rb') as fh:
        return fh.read()

def _compress_group(group, settings):
    cache, results = {}, []
    for path, locator in group:
        data = _read_input(locator, settings)
        results.append(compress_file(path, data, settings, cache))
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

def _executor(jobs, initializer=None, initargs=()):
    return concurrent.futures.ProcessPoolExecutor(max_workers=jobs, mp_context=multiprocessing.get_context('forkserver'), initializer=initializer, initargs=initargs)

class _NoProgress:
    def __init__(self, total):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        pass

    def update(self, n):
        pass

def compress(sources, dest, settings=Settings(), jobs=None, progress=_NoProgress):
    """Compress files under sources into a new container at dest, return the manifest. progress(total) returns a
    context manager with an update(n) method to report progress, e.g. click.progressbar.
    """
    jobs = jobs or default_jobs()
    groups = group_inputs(iter_inputs(sources, settings.extract_archives))
    stats = {
        'files': collections.Counter(),
        'fallbacks': collections.Counter(),
        'dropped': collections.Counter(),
        'uncompressed': collections.Counter(),
        'blobs': 0,
    }
    with Writer(dest, level=settings.level, block_size=settings.block_size, processors=jobs) as writer, \
            progress(sum(map(len, groups))) as bar:
        with _executor(jobs) as executor:
            for results in _imap(executor, functools.partial(_compress_group, settings=settings), groups, ahead=2 * jobs):
                for result in results:
                    stats['uncompressed'].update(result.uncompressed)
                    if result.kind is None:
                        stats['dropped'][classify(result.path)] += 1
                        continue
                    stats['files'][result.kind] += 1
                    xattrs = { container.XATTR_SHA256: result.sha256, container.XATTR_SIZE: str(result.size) }
                    if result.fallback is not None:
                        stats['fallbacks'][result.fallback] += 1
                        xattrs[container.XATTR_FALLBACK] = result.fallback
                    for path, blob in result.blobs:
                        if path not in writer:
                            writer.add(path, blob)
                            stats['blobs'] += 1
                    writer.add(f'contents/{result.kind}/{result.path}', result.stub, xattrs)
                    bar.update(1)
        manifest = {
            'format': container.FORMAT_VERSION,
            'af3io_version': importlib.metadata.version('af3io'),
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
    exists already (unless force) and KeyError if a pattern matches no files.
    """
    jobs = jobs or default_jobs()
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
    with progress(len(names)) as bar:
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
