"""
    Compress AlphaFold 3 inputs/outputs into a single squashfs image (.af3io.sqfs), see design/specs/compress.md

        af3io compress af3io-testdata/mgen_pools_5k_top3 mgen_pools_5k_top3.af3io.sqfs
        af3io decompress mgen_pools_5k_top3.af3io.sqfs .
        af3io info mgen_pools_5k_top3.af3io.sqfs

    Files are read back without decompressing to disk with af3io.compress.open (see Container), or as predictions
    with af3io.predictions (e.g. Predictions('mgen_pools_5k_top3.af3io.sqfs::mgen_pools_5k_top3/pools_5k_0040f80')).
"""
from .container import Container, ChecksumError, classify
from .pipeline import Settings, compress, decompress

def open(path):
    """Open an af3io container for reading"""
    return Container(path)

def is_container(path):
    return str(path).endswith('.af3io.sqfs')
