#!/usr/bin/env python

from pathlib import Path
import allies
import numpy as np
from pathlib import Path
import yaml
from numbers import Number

HERE=Path(allies.__file__).parent

def safe_delete(annotation, segment):
    """deletes segment from annotation if it exists
    If not, warns the user
    """
    if annotation.get_tracks(segment):
        del annotation[segment]
    else:
        print(f'{segment} is not in '
              f'{annotation.uri if annotation.uri else "anonymous annotation"}')

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
    for subset_lst in (HERE/"protocols").iterdir():
        with open(subset_lst) as file:
            subset = set(file.read().split('\n'))
            subset.discard('')
        protocols[subset_lst.stem]=subset
    return protocols

def get_params():
    """Loads ./config.yml in a dict and returns it"""
    with open(HERE/'config.yml') as file:
        params = yaml.load(file)
    return params

def label_generator(references):
    """Yields ascending integers that are not in references"""
    label=0
    labels = references.keys()
    while True:
        label+=1
        if label not in labels:
            yield label

def hypothesis_to_unk(hypothesis):
    """Returns a sub-annotation of hypothesis where all labels are < 0
    Also converts label to `str` because of speech turn clustering pipeline
    See SpeakerIdentification
    """
    unknown = hypothesis.empty()
    for segment, track, label in hypothesis.itertracks(yield_label=True):
        if isinstance(label, Number) and label < 0:
            unknown[segment, track] = str(label)
            del hypothesis[segment, track]
    return unknown, hypothesis
