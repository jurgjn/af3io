"""
    Minimal read-only squashfs (version 4.0) reader: path lookup, directory listings, file contents, extended
    attributes (xattrs), and the on-disk (compressed) size of every file

    Metadata (inodes, directories, fragment/xattr tables) is read on demand and cached, so opening an image and
    reading a single file only touches the blocks on the way to it. Writing images is left to squashfs-tools
    (sqfstar, see af3io.compress.container).

    Format: https://dr-emann.github.io/squashfs/ and squashfs-tools (squashfs_fs.h)
"""
import dataclasses, lzma, os, struct, zlib
try:
    from compression import zstd # Python >= 3.14
except ImportError:
    from backports import zstd

MAGIC = 0x73717368
NONE32 = 0xFFFFFFFF
NONE64 = 0xFFFFFFFFFFFFFFFF
METADATA_SIZE = 8192
BLOCK_UNCOMPRESSED = 1 << 24
FLAG_COMPRESSOR_OPTIONS = 0x0400

# Inode types
DIR, FILE, SYMLINK, LDIR, LFILE, LSYMLINK = 1, 2, 3, 8, 9, 10

_SUPERBLOCK = struct.Struct('<IIIIIHHHHHHQQQQQQQQ')
_XATTR_PREFIX = {0: 'user.', 1: 'trusted.', 2: 'security.'}

def _decompressor(compression_id):
    if compression_id == 6:
        return zstd.decompress
    elif compression_id == 1:
        return zlib.decompress
    elif compression_id == 4:
        return lzma.decompress
    raise NotImplementedError(f'squashfs compressor id {compression_id} not supported (use zstd, gzip or xz)')

@dataclasses.dataclass(frozen=True, slots=True)
class Inode:
    ref: int             # inode reference: metadata block (relative to the inode table) << 16 | offset
    type: int            # basic type: DIR, FILE, SYMLINK (extended types are mapped to basic ones)
    size: int = 0        # file size (directories: size of the listing in the directory table)
    blocks_start: int = 0
    block_sizes: tuple = ()
    fragment: int = NONE32
    fragment_offset: int = 0
    xattr: int = NONE32
    dir_block: int = 0
    dir_offset: int = 0
    target: str = None   # symlink target

    def isdir(self):
        return self.type == DIR

    def isfile(self):
        return self.type == FILE

class SquashFS:
    def __init__(self, path):
        self.path = os.fspath(path)
        self._fd = os.open(self.path, os.O_RDONLY)
        try:
            (magic, self.inode_count, _, self.block_size, self.fragment_count, compression_id, _, self.flags, _,
                major, minor, self.root_ref, self.bytes_used, self._id_table, self._xattr_table, self._inode_table,
                self._dir_table, self._fragment_table, _) = _SUPERBLOCK.unpack(self._pread(0, _SUPERBLOCK.size))
            if magic != MAGIC or (major, minor) != (4, 0):
                raise ValueError(f'{self.path}: not a squashfs 4.0 image')
        except BaseException:
            os.close(self._fd)
            raise
        self._decompress = _decompressor(compression_id)
        self._metadata_cache = {}
        self._fragment_cache = {}
        self._dir_cache = {}
        self._fragment_entries = None
        self._xattr_ids = None
        self.root = self.inode(self.root_ref)

    def close(self):
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()

    def __getstate__(self):
        raise TypeError('SquashFS readers cannot be pickled, open the image in each process instead')

    def _pread(self, offset, size):
        data = os.pread(self._fd, size, offset)
        if len(data) != size:
            raise EOFError(f'{self.path}: unexpected end of image at offset {offset}')
        return data

    # Metadata blocks: 16-bit header (size, bit 15 set if uncompressed) followed by up to 8 KiB of data
    def _metadata_block(self, pos):
        try:
            return self._metadata_cache[pos]
        except KeyError:
            pass
        header, = struct.unpack('<H', self._pread(pos, 2))
        size = header & 0x7FFF
        data = self._pread(pos + 2, size)
        if not (header & 0x8000):
            data = self._decompress(data)
        if len(self._metadata_cache) > 65536: # ~512 MiB of decompressed metadata
            self._metadata_cache.clear()
        self._metadata_cache[pos] = result = (data, pos + 2 + size)
        return result

    def _metadata(self, pos, offset, size):
        # Read size bytes of a metadata stream starting at offset in the block at pos, return (bytes, pos, offset) of the end
        parts = []
        while size > 0:
            data, next_pos = self._metadata_block(pos)
            part = data[offset:offset + size]
            parts.append(part)
            size -= len(part)
            offset += len(part)
            if offset >= len(data):
                pos, offset = next_pos, 0
        return b''.join(parts), pos, offset

    class _Cursor:
        def __init__(self, image, pos, offset):
            self.image, self.pos, self.offset = image, pos, offset

        def read(self, size):
            data, self.pos, self.offset = self.image._metadata(self.pos, self.offset, size)
            return data

        def unpack(self, fmt):
            return struct.unpack(fmt, self.read(struct.calcsize(fmt)))

    def _cursor(self, pos, offset):
        return SquashFS._Cursor(self, pos, offset)

    def inode(self, ref):
        cur = self._cursor(self._inode_table + (ref >> 16), ref & 0xFFFF)
        type_, _mode, _uid, _gid, _mtime, _number = cur.unpack('<HHHHII')
        if type_ == DIR:
            dir_block, _nlink, size, dir_offset, _parent = cur.unpack('<IIHHI')
            return Inode(ref, DIR, size=size, dir_block=dir_block, dir_offset=dir_offset)
        elif type_ == LDIR:
            _nlink, size, dir_block, _parent, _index_count, dir_offset, xattr = cur.unpack('<IIIIHHI')
            return Inode(ref, DIR, size=size, dir_block=dir_block, dir_offset=dir_offset, xattr=xattr)
        elif type_ in (FILE, LFILE):
            if type_ == FILE:
                blocks_start, fragment, fragment_offset, size = cur.unpack('<IIII')
                xattr = NONE32
            else:
                blocks_start, size, _sparse, _nlink, fragment, fragment_offset, xattr = cur.unpack('<QQQIIII')
            n_blocks = size // self.block_size if fragment != NONE32 else -(-size // self.block_size)
            block_sizes = cur.unpack(f'<{n_blocks}I')
            return Inode(ref, FILE, size=size, blocks_start=blocks_start, block_sizes=block_sizes,
                fragment=fragment, fragment_offset=fragment_offset, xattr=xattr)
        elif type_ in (SYMLINK, LSYMLINK):
            _nlink, target_size = cur.unpack('<II')
            target = cur.read(target_size).decode()
            xattr = cur.unpack('<I')[0] if type_ == LSYMLINK else NONE32
            return Inode(ref, SYMLINK, size=target_size, xattr=xattr, target=target)
        else: # devices, fifos, sockets
            return Inode(ref, type_)

    def listdir(self, inode):
        """Directory entries of inode as {name: inode reference}, in sorted order"""
        if not inode.isdir():
            raise NotADirectoryError(inode)
        try:
            return self._dir_cache[inode.ref]
        except KeyError:
            pass
        entries = {}
        cur = self._cursor(self._dir_table + inode.dir_block, inode.dir_offset)
        remaining = inode.size - 3 # listing size includes 3 bytes for the implicit . and .. entries
        while remaining > 0:
            count, start, _number = cur.unpack('<III')
            remaining -= 12
            for _ in range(count + 1):
                offset, _number_delta, _type, name_size = cur.unpack('<HhHH')
                name = cur.read(name_size + 1).decode('utf-8', errors='surrogateescape')
                entries[name] = (start << 16) | offset
                remaining -= 8 + name_size + 1
        if len(self._dir_cache) > 65536:
            self._dir_cache.clear()
        self._dir_cache[inode.ref] = entries
        return entries

    def lookup(self, path):
        """Inode at path (absolute or relative to the root), or None if it does not exist"""
        inode = self.root
        for name in path.strip('/').split('/'):
            if name == '':
                continue
            if not inode.isdir():
                return None
            ref = self.listdir(inode).get(name)
            if ref is None:
                return None
            inode = self.inode(ref)
        return inode

    def walk(self, path='/'):
        """Yield (path, inode) of all regular files under path (depth-first, sorted by name)"""
        inode = self.lookup(path)
        if inode is None:
            return
        if inode.isfile():
            yield path, inode
            return
        stack = [ (path.rstrip('/'), iter(self.listdir(inode).items())) ]
        while stack:
            dir_path, entries = stack[-1]
            for name, ref in entries:
                child = self.inode(ref)
                if child.isdir():
                    stack.append((f'{dir_path}/{name}', iter(self.listdir(child).items())))
                    break
                elif child.isfile():
                    yield f'{dir_path}/{name}', child
            else:
                stack.pop()

    def _fragment_entry(self, index):
        # Fragment table: u64 pointers to metadata blocks with 16-byte entries (start u64, size u32, unused u32)
        if self._fragment_entries is None:
            n_blocks = -(-self.fragment_count * 16 // METADATA_SIZE)
            self._fragment_entries = struct.unpack(f'<{n_blocks}Q', self._pread(self._fragment_table, 8 * n_blocks))
        pos = self._fragment_entries[index * 16 // METADATA_SIZE]
        start, size, _ = self._cursor(pos, index * 16 % METADATA_SIZE).unpack('<QII')
        return start, size

    def _fragment_block(self, index):
        try:
            return self._fragment_cache[index]
        except KeyError:
            pass
        data = self._block(*self._fragment_entry(index))
        if len(self._fragment_cache) >= 16:
            self._fragment_cache.pop(next(iter(self._fragment_cache)))
        self._fragment_cache[index] = data
        return data

    def _block(self, start, size):
        data = self._pread(start, size & ~BLOCK_UNCOMPRESSED)
        return data if size & BLOCK_UNCOMPRESSED else self._decompress(data)

    def read(self, inode):
        """Contents of a regular file as bytes"""
        if isinstance(inode, str):
            path, inode = inode, self.lookup(inode)
            if inode is None:
                raise FileNotFoundError(path)
        if not inode.isfile():
            raise IsADirectoryError(inode) if inode.isdir() else ValueError(f'not a regular file: {inode}')
        parts = []
        pos = inode.blocks_start
        remaining = inode.size
        for block_size in inode.block_sizes:
            disk_size = block_size & ~BLOCK_UNCOMPRESSED
            if disk_size == 0: # sparse block
                parts.append(bytes(min(self.block_size, remaining)))
            else:
                parts.append(self._block(pos, block_size))
                pos += disk_size
            remaining -= len(parts[-1])
        if inode.fragment != NONE32 and remaining > 0:
            parts.append(self._fragment_block(inode.fragment)[inode.fragment_offset:inode.fragment_offset + remaining])
        data = b''.join(parts)
        if len(data) != inode.size:
            raise ValueError(f'{self.path}: size mismatch reading inode {inode.ref:#x}')
        return data

    def xattrs(self, inode):
        """Extended attributes of inode as {name: value (bytes)}, e.g. {'user.af3io.sha256': b'...'}"""
        if inode.xattr == NONE32 or self._xattr_table == NONE64:
            return {}
        if self._xattr_ids is None:
            kv_start, n_ids, _ = struct.unpack('<QII', self._pread(self._xattr_table, 16))
            n_blocks = -(-n_ids * 16 // METADATA_SIZE)
            self._xattr_ids = (kv_start, struct.unpack(f'<{n_blocks}Q', self._pread(self._xattr_table + 16, 8 * n_blocks)))
        kv_start, id_blocks = self._xattr_ids
        ref, count, _size = self._cursor(id_blocks[inode.xattr * 16 // METADATA_SIZE], inode.xattr * 16 % METADATA_SIZE).unpack('<QII')
        cur = self._cursor(kv_start + (ref >> 16), ref & 0xFFFF)
        xattrs = {}
        for _ in range(count):
            type_, name_size = cur.unpack('<HH')
            name = _XATTR_PREFIX.get(type_ & 0xFF, '') + cur.read(name_size).decode()
            value_size, = cur.unpack('<I')
            value = cur.read(value_size)
            if type_ & 0x100: # value stored out of line, value is a reference to it
                value_ref, = struct.unpack('<Q', value)
                value_cur = self._cursor(kv_start + (value_ref >> 16), value_ref & 0xFFFF)
                value = value_cur.read(value_cur.unpack('<I')[0])
            xattrs[name] = value
        return xattrs

    def disk_usage(self, files):
        """On-disk (compressed) bytes for each of files, an iterable of (key, inode): {key: bytes}
        Data shared between files (duplicates removed by mksquashfs) is counted once, for the first file;
        fragment blocks (tail ends of several files) are split in proportion to the uncompressed tail sizes.
        """
        usage, seen, fragments = {}, set(), {}
        for key, inode in files:
            identity = (inode.blocks_start, inode.block_sizes, inode.fragment, inode.fragment_offset, inode.size)
            if identity in seen:
                usage[key] = 0
                continue
            seen.add(identity)
            usage[key] = sum(size & ~BLOCK_UNCOMPRESSED for size in inode.block_sizes)
            if inode.fragment != NONE32:
                tail = inode.size - self.block_size * len(inode.block_sizes)
                fragments.setdefault(inode.fragment, []).append((key, tail))
        for index, tails in fragments.items():
            disk_size = self._fragment_entry(index)[1] & ~BLOCK_UNCOMPRESSED
            total = sum(tail for _, tail in tails)
            for key, tail in tails:
                usage[key] += disk_size * tail / total if total > 0 else 0
        return usage
