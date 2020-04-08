#!/usr/bin/env python

def print_stats(stats):
    """Pretty-prints protocol statistics"""
    for key,value in stats.items():
        if key=='labels':
            print(f'n_speakers: {len(value)}')
        else:
            print(f'{key}: {value:.0f}')
