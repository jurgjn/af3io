
from itertools import chain, repeat

def decompress_repeating(s):
    return list(eval(s, globals={'__builtins__': None}, locals={'chain': chain, 'repeat': repeat}))

def decompress_increasing(s):
    return list(eval(s, globals={'__builtins__': None}, locals={'chain': chain, 'range': range}))
