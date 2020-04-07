#!/usr/bin/env python

from pyannote.core import Annotation, Segment, Timeline
from warnings import warn
from pyannote.database.protocol import SpeakerDiarizationProtocol

def AlliesProtocol(SpeakerDiarizationProtocol):

    def trn_iter(self):
        raise NotImplementedError(
            'Custom speaker diarization protocol should implement "trn_iter".')

    def dev_iter(self):
        raise NotImplementedError(
            'Custom speaker diarization protocol should implement "dev_iter".')

    def tst_iter(self):
        raise NotImplementedError(
            'Custom speaker diarization protocol should implement "tst_iter".')

def uem_to_timeline(uem, uri = None):
    """Converts Allies-UEM to pyannote Timeline

    Parameters:
    -----------
    uem: dict or uemranges
      - 'start_time': list[float], start times of speech in seconds
      - 'end_time': list[float], end times of speech in seconds
    uri : str, optional
        Uniform Resource Identifier of the Timeline.
        Defaults to None.

    Returns:
    --------
    timeline, see pyannote.core.Timeline
    """
    if isinstance(uem,dict):
        uem=zip(uem['start_time'], uem['end_time'])
    else:
        uem=zip(uem.start_time,uem.end_time)
    segments = [Segment(start, end) for start, end in uem]
    return Timeline(segments, uri)

def speakers_to_annotation(speakers, uri = None, modality = 'speaker'):
    annotation = Annotation(uri, modality)
    for idx in range(speakers.speaker.size):
        start=speakers.start_time[idx]
        stop=speakers.end_time[idx]
        segment = Segment(start, stop)
        annotation[segment, idx] = speakers.speaker[idx]
    return annotation


def load_train(data_loaders):
    """Loads (initial) training set
    i.e. :
    - `speakers` in `diar_train`
    - `train_speakers` in `diar_lifelong`
    """
    print('loading train data')

    # The loader is the interface used to acces inputs of this algorithmic block
    loader = data_loaders.loaderOf("features")

    # create pyannote protocol
    # TODO (how ?)
    protocol = {}
    for i in range(loader.count()):
        (data, _, end_index) = loader[i]
        speakers = data["speakers"]
        file_id = data["file_info"].file_id
        supervision = data["file_info"].supervision
        time_stamp = data["file_info"].time_stamp
        uem = data["uem"]
        annotated = uem_to_timeline(uem, file_id)
        annotation = speakers_to_annotation(speakers, file_id).crop(annotated)
        current_file = {
            'annotation' : annotation,
            'annotated' : annotated
        }
        protocol[file_id] = current_file

    # TODO handle features (i.e. waveform)
    # TODO create ProtocolFile

    return protocol
