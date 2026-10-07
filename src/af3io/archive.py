"""
    Traverse and read files inside nested archives (tar, zip) with optional compression (gzip, zstd)

    A file inside (nested) archives is addressed by a locator string, which joins the on-disk path and the
    member names, outermost first, with '::', e.g.:
        pools_5k.tar::pools_5k_0040f80.zip::pools_5k_0040f80/pools_5k_0040f80_model.cif

    Compression is detected from the file name and decompressed transparently:
        .gz / .zst / .zstd     single compressed file (e.g. foo.zip.gz, foo.json.zst)
        .tgz / .tzst           shorthand for .tar.gz / .tar.zst
    Containers (descended into by walk) are detected from the file name after removing compression suffixes:
        .tar / .zip

    Random access is fast for uncompressed tar files on disk (member offsets are cached per process) and for
    zip files. Compressed tar files (.tar.gz, .tar.zst) can only be read sequentially, so every open() of one
    of their members re-reads the archive from the start; prefer walk() or uncompressed tar files instead.
"""
import builtins, contextlib, functools, gzip, io, os, tarfile, zipfile
try:
    from compression import zstd # Python >= 3.14
except ImportError:
    from backports import zstd
from pathlib import PurePosixPath

SEP = '::'

def join(*parts):
    return SEP.join(os.fspath(part) for part in parts)

def split(locator):
    return os.fspath(locator).split(SEP)

def is_nested(locator):
    return SEP in os.fspath(locator)

def relpath(locator, start=os.curdir):
    """Locator with the on-disk path relative to start (default: current directory), e.g.
    /data/pools_5k.tar::pools_5k_0040f80.zip => pools_5k.tar::pools_5k_0040f80.zip if run from /data
    """
    path, *members = split(locator)
    return join(os.path.relpath(path, start), *members)

# suffix => (replacement suffix, decompressor)
def _gzip_open(fh):
    return gzip.GzipFile(fileobj=fh, mode='rb')

def _zstd_open(fh):
    return zstd.ZstdFile(fh, mode='rb')

_COMPRESSION = {
    '.gz':   ('',     _gzip_open),
    '.tgz':  ('.tar', _gzip_open),
    '.zst':  ('',     _zstd_open),
    '.zstd': ('',     _zstd_open),
    '.tzst': ('.tar', _zstd_open),
}

def strip_compression(name):
    """Name with compression suffixes removed, e.g. dir/foo.tar.gz => dir/foo.tar, foo.tgz => foo.tar"""
    name = os.fspath(name)
    while (suffix := PurePosixPath(name).suffix.lower()) in _COMPRESSION:
        name = name[:-len(suffix)] + _COMPRESSION[suffix][0]
    return name

def _decompress(stack, fh, name, random_access):
    # Apply decompression layers based on name; decompressed streams only support sequential reads
    while (suffix := PurePosixPath(name).suffix.lower()) in _COMPRESSION:
        replacement, open_ = _COMPRESSION[suffix]
        fh = stack.enter_context(open_(fh))
        name = name[:-len(suffix)] + replacement
        random_access = False
    return fh, name, random_access

@contextlib.contextmanager
def decompress(fh, name):
    """Decompress an open binary file object based on its name (no-op if name has no compression suffix)"""
    with contextlib.ExitStack() as stack:
        yield _decompress(stack, fh, name, True)[0]

def _container_type(name):
    suffix = PurePosixPath(name).suffix.lower()
    return suffix if suffix in ('.tar', '.zip') else None

@functools.lru_cache(maxsize=16)
def _tar_index(path, mtime_ns, size):
    # Member name => TarInfo for an uncompressed tar file on disk (scanned once per process)
    with tarfile.open(path, mode='r:') as tf:
        return { member.name: member for member in tf.getmembers() }

def _zip_member_random_access(info):
    # Seeking backwards in deflated members re-decompresses from the start; buffer these instead
    return info.compress_type == zipfile.ZIP_STORED

def walk(path):
    """Yield locators of all regular files under path (a directory, an archive, or a plain file), descending
    into directories, (nested) archives and compressed files. Directories are traversed in sorted order,
    archives in archive order.
    """
    path = os.fspath(path)
    if os.path.isdir(path):
        for root, dirs, files in os.walk(path):
            dirs.sort()
            for file in sorted(files):
                yield from walk(os.path.join(root, file))
        return

    if _container_type(strip_compression(path)) is None: # plain or compressed file, no need to open
        yield path
        return

    with contextlib.ExitStack() as stack:
        fh = stack.enter_context(builtins.open(path, 'rb'))
        yield from _walk_fileobj(stack, path, path, fh, random_access=True)

def _walk_fileobj(stack, locator, name, fh, random_access):
    fh, name, random_access = _decompress(stack, fh, name, random_access)
    container = _container_type(name)
    if container == '.tar':
        with tarfile.open(fileobj=fh, mode='r:' if random_access else 'r|') as tf:
            for member in tf:
                if member.isfile():
                    with contextlib.ExitStack() as member_stack:
                        member_fh = member_stack.enter_context(tf.extractfile(member))
                        yield from _walk_fileobj(member_stack, join(locator, member.name), member.name, member_fh, random_access)
    elif container == '.zip':
        if not random_access:
            fh = io.BytesIO(fh.read())
        with zipfile.ZipFile(fh) as zf:
            for info in zf.infolist():
                if not info.is_dir():
                    with contextlib.ExitStack() as member_stack:
                        member_fh = member_stack.enter_context(zf.open(info))
                        yield from _walk_fileobj(member_stack, join(locator, info.filename), info.filename, member_fh, _zip_member_random_access(info))
    else:
        yield locator

@contextlib.contextmanager
def open(locator, seekable=False):
    """Open a (possibly nested and/or compressed) file for reading as a binary file object, e.g.
    open('pools_5k.tar::pools_5k_0040f80.zip'). Compressed files are decompressed transparently.
    With seekable=True, the result is buffered in memory if it does not support efficient random access.
    """
    path, *members = split(locator)
    with contextlib.ExitStack() as stack:
        fh = stack.enter_context(builtins.open(path, 'rb'))
        fh, name, random_access = _decompress(stack, fh, path, True)
        for depth, member in enumerate(members):
            container = _container_type(name)
            if container == '.tar' and random_access:
                tf = stack.enter_context(tarfile.open(fileobj=fh, mode='r:'))
                if depth == 0: # uncompressed tar on disk, use cached member index
                    stat = os.stat(path)
                    info = _tar_index(os.path.abspath(path), stat.st_mtime_ns, stat.st_size)[member]
                else:
                    info = tf.getmember(member)
                fh = stack.enter_context(tf.extractfile(info))
            elif container == '.tar':
                tf = stack.enter_context(tarfile.open(fileobj=fh, mode='r|'))
                for info in tf:
                    if info.name == member:
                        break
                else:
                    raise KeyError(f'{member} not found in {join(path, *members[:depth])}')
                fh = stack.enter_context(tf.extractfile(info))
            elif container == '.zip':
                if not random_access:
                    fh = io.BytesIO(fh.read())
                zf = stack.enter_context(zipfile.ZipFile(fh))
                info = zf.getinfo(member)
                fh = stack.enter_context(zf.open(info))
                random_access = _zip_member_random_access(info)
            else:
                raise ValueError(f'{join(path, *members[:depth])} is not a tar or zip archive')
            fh, name, random_access = _decompress(stack, fh, member, random_access)

        if seekable and not random_access:
            fh = io.BytesIO(fh.read())
        yield fh
