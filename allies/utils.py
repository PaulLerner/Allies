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

def hypothesis_to_unk(hypothesis):
    """Returns a sub-annotation of hypothesis where all labels are < 0
    See SpeakerIdentification
    """
    unknown_labels = [label for label in hypothesis.labels()
                      if isinstance(label, Number) and label < 0]
    unknown = hypothesis.subset(unknown_labels, invert = False)
    hypothesis = hypothesis.subset(unknown_labels, invert = True)
    return unknown, hypothesis

def relabel_unknown(hypothesis):
    """Relabels unknown segments (i.e. with a `Number` label) with a unique label"""
    for i, (segment, track, label) in enumerate(hypothesis.itertracks(yield_label=True)):
        if isinstance(label, Number):
            hypothesis[segment, track] = i