#!/usr/bin/env python

from pathlib import Path
import allies
import numpy as np
from pathlib import Path
import yaml
from numbers import Number
from typing import Text

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

def mutual_cl(hypothesis):
    """Mutually cannot-link identified speakers (tagged with a `Text` label)

    Parameters
    ----------
    hypothesis: Annotation
        hypothesis with identified speakers tagged with a `Text` label

    Returns
    -------
    cannot_link: dict
        Clustering constraints, a dict like:
        {Segment : Set[Segment]}, where segments should not be clustered together
    """

    cannot_link = {}
    for label in hypothesis.labels():
        if not isinstance(label, Text):
            continue
        timeline = hypothesis.label_timeline(label, copy=False)
        for segment in timeline:
            cannot_link.setdefault(segment, set())
            # get all other segments from other identified speakers
            for other_speaker in hypothesis.labels():
                if not isinstance(other_speaker, Text) or other_speaker == label:
                    continue
                other_timeline = hypothesis.label_timeline(other_speaker, copy=False)
                cannot_link[segment].update(other_timeline.segments_set_)

    return cannot_link

def relabel_unknown(hypothesis):
    """Relabels unknown segments (i.e. with a `Number` label) with a unique label

    e.g.
    >>> print(annotation)
    [ 00:00:00.000 -->  00:00:01.000] _ bob
    [ 00:00:01.000 -->  00:00:02.000] _ 0
    [ 00:00:02.000 -->  00:00:03.000] _ 0

    >>> print(relabel_unknown(annotation))
    [ 00:00:00.000 -->  00:00:01.000] _ bob
    [ 00:00:01.000 -->  00:00:02.000] _ 1
    [ 00:00:02.000 -->  00:00:03.000] _ 2

    """
    for i, (segment, track, label) in enumerate(hypothesis.itertracks(yield_label=True)):
        if isinstance(label, Number):
            hypothesis[segment, track] = i
    return hypothesis