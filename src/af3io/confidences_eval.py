
from itertools import chain

def repeating_decode(s):
    return list(eval(s, {'__builtins__': None}, {'__builtins__': None}))

def increasing_decode(s):
    return list(eval(s, {'__builtins__': None}, {'chain': chain, 'range': range}))
