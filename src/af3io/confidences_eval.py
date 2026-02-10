
from itertools import chain, repeat

def decompress_repeating(s):
    return list(eval(s, {'__builtins__': None}, {'chain': chain, 'repeat': repeat}))

def decompress_increasing(s):
    return list(eval(s, {'__builtins__': None}, {'chain': chain, 'range': range}))
