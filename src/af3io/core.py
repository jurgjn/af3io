
import gzip

def _open_r(path):
    try: # AttributeError: '_io.TextIOWrapper' object has no attribute 'endswith' 
        if path.endswith('.gz'):
            return gzip.open(path, 'rt')
        else:
            return open(path, 'r')
    except:
        return path

def uf(x):
    return '{:,}'.format(x)

def ul(x):
    return uf(len(x))

__all__ = ['_open_r', 'uf', 'ul']
