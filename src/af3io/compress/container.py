"""
    af3io containers (.af3io): squashfs images with the following layout

        /af3io_manifest.json                    af3io version, compression settings and totals
        /contents/<kind>/<path>                 one file for every input file at <path>, kind is one of KINDS
        /blobs/<kind>/<field>/<ab>/<sha256>     parts extracted from files under /contents, e.g. PAE matrices

    Files under /contents carry the sha256 and size of their original contents as extended attributes (xattrs).
    Blobs are named after the sha256 of the exact bytes they replace (<ab> is the first two characters of it, as in
    git), and referenced from files under /contents as af3io:///blobs/<kind>/<field>/<ab>/<sha256>.

    Containers are written with sqfstar from squashfs-tools (a tar stream with xattrs in PAX headers) and read
    with af3io.compress.squashfs.
"""
import collections, hashlib, io, json, os, posixpath, shutil, subprocess, tarfile, tempfile

import pandas as pd

from .squashfs import SquashFS

FORMAT_VERSION = 1
MANIFEST = 'af3io_manifest.json'
KINDS = ('data', 'confidences', 'model', 'other')
REF_SCHEME = 'af3io://'
XATTR_SHA256 = 'user.af3io.sha256'
XATTR_SIZE = 'user.af3io.size'
XATTR_FALLBACK = 'user.af3io.fallback'

# Size categories reported by info(), see category()
CATEGORIES = ('data', 'model', 'pae', 'contact_probs', 'confidences', 'other', 'squashfs')

class ChecksumError(ValueError):
    pass

def classify(path):
    """Kind of a file from its name: data pipeline output (*_data.json), confidences (*confidences.json apart
    from *summary_confidences.json), predicted structure (*model.cif), or other (see af3io.predictions.Predictions
    for AlphaFold 3 output naming patterns)
    """
    name = posixpath.basename(path)
    if name.endswith('_data.json'):
        return 'data'
    elif name.endswith('confidences.json') and not name.endswith('summary_confidences.json'):
        return 'confidences'
    elif name.endswith('model.cif'):
        return 'model'
    return 'other'

def category(path):
    """Size category of a path in the container, e.g. /blobs/confidences/pae/ab/abcd... => pae"""
    parts = path.strip('/').split('/')
    if parts[0] == 'contents' and len(parts) > 2:
        return parts[1]
    elif parts[0] == 'blobs' and len(parts) > 2:
        return parts[2] if parts[1] == 'confidences' else parts[1]
    return 'squashfs'

def sha256(data):
    return hashlib.sha256(data).hexdigest()

def blob_path(kind, field, digest):
    return f'blobs/{kind}/{field}/{digest[:2]}/{digest}'

def blob_ref(path):
    return f'{REF_SCHEME}/{path}'

def codec(kind):
    """Codec for files of a kind (see af3io.compress.codec), None if there is none"""
    if kind == 'data':
        from . import data
        return data
    elif kind == 'confidences':
        from . import confidences
        return confidences
    return None

class Writer:
    """Write a new container by streaming a tar file with xattrs in PAX headers into sqfstar. The image is written
    to a temporary directory next to path, and moved into place by close().
    """
    def __init__(self, path, level=19, block_size=1 << 20, processors=None):
        self.path = os.fspath(path)
        if os.path.lexists(self.path):
            raise FileExistsError(f'{self.path} already exists')
        self.level, self.block_size = level, block_size
        self._tmpdir = tempfile.mkdtemp(prefix=f'.{os.path.basename(self.path)}.', dir=os.path.dirname(os.path.abspath(self.path)))
        self._image = os.path.join(self._tmpdir, 'image.af3io')
        self._log = open(os.path.join(self._tmpdir, 'sqfstar.log'), 'w+b')
        cmd = ['sqfstar', '-comp', 'zstd', '-Xcompression-level', str(level), '-b', str(block_size),
            '-reproducible', '-mkfs-time', '0', '-all-time', '0', '-all-root', '-default-mode', '0755',
            '-quiet', '-no-progress', *(['-processors', str(processors)] if processors else []), self._image]
        try:
            self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=self._log, stderr=self._log)
        except BaseException:
            self._cleanup()
            raise
        self._tar = tarfile.open(fileobj=self._proc.stdin, mode='w|', format=tarfile.PAX_FORMAT)
        self._paths = set()

    def __contains__(self, path):
        return path in self._paths

    def add(self, path, data, xattrs=None):
        """Add a file at path (relative to the root of the image) with xattrs {name: str}"""
        if path in self._paths:
            raise ValueError(f'{path} already added')
        self._paths.add(path)
        info = tarfile.TarInfo(path)
        info.size, info.mode, info.mtime = len(data), 0o644, 0
        info.pax_headers = { f'SCHILY.xattr.{name}': value for name, value in (xattrs or {}).items() }
        try:
            self._tar.addfile(info, io.BytesIO(data))
        except BrokenPipeError:
            self._proc.wait()
            raise RuntimeError(f'sqfstar exited early: {self._read_log()}')

    def _read_log(self):
        self._log.seek(0)
        return self._log.read().decode(errors='replace').strip()

    def _cleanup(self):
        self._log.close()
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def close(self):
        try:
            self._tar.close()
            self._proc.stdin.close()
        except BrokenPipeError:
            pass
        if self._proc.wait() != 0:
            log = self._read_log()
            self._cleanup()
            raise RuntimeError(f'sqfstar failed: {log}')
        try:
            os.link(self._image, self.path) # unlike rename, fails if path was created in the meantime
        finally:
            self._cleanup()

    def abort(self):
        self._proc.kill()
        self._proc.wait()
        self._cleanup()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            self.close()
        else:
            self.abort()

class Container:
    """Read files from an af3io container. Files are re-generated in memory from their parts and verified against
    the sha256 of the original contents.
    """
    def __init__(self, path):
        self.path = os.fspath(path)
        self.fs = SquashFS(self.path)
        try:
            manifest = self.fs.lookup(f'/{MANIFEST}')
            if manifest is None:
                raise ValueError(f'{self.path}: not an af3io container (no {MANIFEST})')
            self.manifest = json.loads(self.fs.read(manifest))
            if self.manifest.get('format') != FORMAT_VERSION:
                raise ValueError(f'{self.path}: unsupported af3io container format {self.manifest.get("format")}')
        except BaseException:
            self.fs.close()
            raise

    def close(self):
        self.fs.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()

    def _lookup(self, name):
        name = name.strip('/')
        for kind in dict.fromkeys([classify(name), 'other']):
            inode = self.fs.lookup(f'/contents/{kind}/{name}')
            if inode is not None and inode.isfile():
                return kind, inode
        raise KeyError(name)

    def entries(self, subpath=''):
        """Yield (name, kind, inode) of all files under subpath (a directory or a single file), sorted by name"""
        entries = []
        for kind in KINDS:
            prefix = f'/contents/{kind}/'
            for path, inode in self.fs.walk(prefix + subpath.strip('/')):
                entries.append((path[len(prefix):], kind, inode))
        return sorted(entries, key=lambda entry: entry[0])

    def namelist(self, subpath=''):
        """Names (original paths) of all files under subpath"""
        return [ name for name, _, _ in self.entries(subpath) ]

    def __contains__(self, name):
        try:
            self._lookup(name)
            return True
        except KeyError:
            return False

    def read(self, name, verify=True):
        """Contents of the file name as bytes, re-generated from its parts and verified against its sha256"""
        kind, inode = self._lookup(name)
        return self._read(name, kind, inode, verify)

    def _read(self, name, kind, inode, verify=True):
        data = self.fs.read(inode)
        codec_ = codec(kind)
        if codec_ is not None:
            data = codec_.join(data, self.read_blob)
        if verify:
            self._verify(name, inode, data)
        return data

    def _verify(self, name, inode, data):
        expected = self.fs.xattrs(inode).get(XATTR_SHA256)
        if expected is None:
            raise ChecksumError(f'{self.path}: no checksum for {name}')
        if sha256(data) != expected.decode():
            raise ChecksumError(f'{self.path}: checksum mismatch for {name}')

    def open(self, name, verify=True):
        """Contents of the file name as a binary file object"""
        return io.BytesIO(self.read(name, verify))

    def iter(self, subpath='', verify=True):
        """Yield (name, contents) of all files under subpath"""
        for name, kind, inode in self.entries(subpath):
            yield name, self._read(name, kind, inode, verify)

    def read_confidences(self, name, verify=True):
        """Confidences JSON as a dict with PAE and contact probabilities as numpy arrays (as np.array(json.load(...)[key])
        would give), decoded from their blobs without parsing them as text, which is faster than json.load(open(name))
        """
        from . import confidences
        kind, inode = self._lookup(name)
        if kind != 'confidences':
            return confidences.read(self._read(name, kind, inode, verify), None)
        checksum = self.fs.xattrs(inode).get(XATTR_SHA256, b'').decode() if verify else None
        try:
            return confidences.read(self.fs.read(inode), lambda path: self.fs.read(f'/{path}'), checksum)
        except ChecksumError:
            raise ChecksumError(f'{self.path}: checksum mismatch for {name}')

    def read_blob(self, path):
        """Raw (decoded) contents of a blob, e.g. blobs/confidences/pae/ab/abcd..."""
        path = path.removeprefix(REF_SCHEME).strip('/')
        _, kind, field, *_ = path.split('/')
        return codec(kind).decode_blob(field, self.fs.read(f'/{path}'))

    def info(self):
        """Uncompressed (original) and compressed (on-disk) bytes by size category as a data frame"""
        files = list(self.fs.walk('/'))
        compressed = collections.Counter()
        for path, size in self.fs.disk_usage(files).items():
            compressed[category(path)] += size
        compressed['squashfs'] += os.path.getsize(self.path) - sum(compressed.values())
        n_files = collections.Counter(category(path) for path, _ in files if path.startswith('/contents/'))
        uncompressed = self.manifest['stats']['uncompressed']
        df = pd.DataFrame({
            'files': [ n_files.get(cat, 0) for cat in CATEGORIES ],
            'uncompressed': [ uncompressed.get(cat, 0) for cat in CATEGORIES ],
            'compressed': [ round(compressed.get(cat, 0)) for cat in CATEGORIES ],
        }, index=pd.Index(CATEGORIES, name='category'))
        df.loc['total'] = df.sum()
        return df
