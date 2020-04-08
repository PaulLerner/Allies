#!/usr/bin/env python

from pyannote.core import Annotation, Segment, Timeline
from warnings import warn
from pyannote.database.protocol import SpeakerDiarizationProtocol
from pyannote.database.protocol.protocol import ProtocolFile

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

def load_protocol(file_generator) -> SpeakerDiarizationProtocol:
        """Given a ProtocolFile generator, instantiate a SpeakerDiarizationProtocol
        which yields the file in the relevant subset_iter

        Parameters
        ----------
        file_generator : generator
            yields ProtocolFile

        Returns
        -------
        protocol : SpeakerDiarizationProtocol instance
        """
        print("loading data in protocol")
        class AlliesProtocol(SpeakerDiarizationProtocol):

            def trn_iter(self):
                for current_file in file_generator:
                    yield current_file

            def dev_iter(self):
                raise NotImplementedError()

            def tst_iter(self):
                raise NotImplementedError()

        return AlliesProtocol()

def yield_train(data_loaders):
    """Yields (initial) training set
    i.e. :
    - `speakers` in `diar_train`
    - `train_speakers` in `diar_lifelong`
    """
    print('loading train data')

    # The loader is the interface used to acces inputs of this algorithmic block
    loader = data_loaders.loaderOf("features")
    for i in range(loader.count()):
        (data, _, end_index) = loader[i]
        speakers = data["speakers"]
        file_id = data["file_info"].file_id
        supervision = data["file_info"].supervision
        time_stamp = data["file_info"].time_stamp
        uem = data["uem"]
        features = data['features'].data.value
        if len(features.shape) != 2:
            print(features.shape)
            msg = (
                f'Precomputed waveform should be provided as a '
                f'(n_samples, n_channels) `np.ndarray`.'
            )
            print(msg)
            raise ValueError(msg)
        annotated = uem_to_timeline(uem, file_id)
        annotation = speakers_to_annotation(speakers, file_id).crop(annotated)
        current_file = {
            'uri' : file_id,
            'annotation' : annotation,
            'annotated' : annotated,
            'waveform' : features
        }
        yield ProtocolFile(current_file)
