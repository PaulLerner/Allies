#!/usr/bin/env python
from pyannote.core import Segment, Timeline


class UEM:
    """
    A wrapper of an ALLIES UEM dict.
    Provides handy methods to work with pyannote.
    """

    def __init__(self, uem):
        self.uem = uem

    def to_timeline(self, uri = None):
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
                    in zip(self.uem['start_time'], self.uem['end_time'])]
        return Timeline(segments, uri)

    def to_annotation(self, uri = None):
        """Converts Allies-UEM to pyannote Annotation

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
        annotation with string labels, see pyannote.core.Annotation
        """
        return self.to_timeline(uri).to_annotation(generator='string')


class AlliesAnnotation:
    """
    A wrapper for pyannote Annotation objects.
    Provides handy methods to work with ALLIES types.
    """

    def __init__(self, annotation):
        self.annotation = annotation

    def to_hypothesis(self):
        """
        Transform this `Annotation` into a `anthony_larcher/speakers/1`
        :return: a hypothesis dict:
          - 'speaker': list[str] list of predicted speaker IDs
          - 'start_time': list[float], start times of speakers in seconds
          - 'end_time': list[float], end times of speakers in seconds
        """
        speakers, start_times, end_times = [], [], []
        for segment, _, label in self.annotation.itertracks(yield_labels=True):
            speakers.append(label)
            start_times.append(segment.start)
            end_times.append(segment.end)
        return {"speaker": speakers,
                "start_time": start_times,
                "end_time": end_times}

