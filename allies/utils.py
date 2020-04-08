#!/usr/bin/env python

from pathlib import Path
import allies
import numpy as np

def print_stats(stats):
    """Pretty-prints protocol statistics"""
    for key,value in stats.items():
        if key=='labels':
            print(f'n_speakers: {len(value)}')
        else:
            print(f'{key}: {value:.0f}')

def get_protocols():
    """Reads ./protocols/*lst and returns file uris in the corresponding subset"""
    protocols={}
    HERE=Path(allies.__file__).parent
    for subset_lst in (HERE/"protocols").iterdir():
        with open(subset_lst) as file:
            subset = set(file.read().split('\n'))
            subset.discard('')
        protocols[subset_lst.stem]=subset
    return protocols
