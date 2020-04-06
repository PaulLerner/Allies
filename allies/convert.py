#!/usr/bin/env python

from pyannote.core import Segment, Timeline

def uem_to_timeline(uem, uri = None):
    """Converts Allies-UEM to pyannote Timeline

    Parameters:
    -----------
    uem: dict of
      - 'start_time': list[float], start times of speech in seconds
      - 'end_time': list[float], end times of speech in seconds
    uri : str, optional
        Uniform Resource Identifier of the Timeline.
        Defaults to None.

    Returns:
    --------
    timeline, see pyannote.core.Timeline
    """
    segments = [Segment(start, end)
                       for start, end
                       in zip(uem['start_time'], uem['end_time'])]
    return Timeline(segments, uri)
