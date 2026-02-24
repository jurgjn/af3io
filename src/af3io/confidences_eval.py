
from itertools import chain

def decompress_repeating(s):
    return list(eval(s, {'__builtins__': None}, {'__builtins__': None}))

def decompress_increasing(s):
    return list(eval(s, {'__builtins__': None}, {'chain': chain, 'range': range}))
