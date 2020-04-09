#!/usr/bin/env python

from pyannote.core import Annotation, Segment, Timeline
from warnings import warn
from pyannote.database.protocol import SpeakerDiarizationProtocol
from pyannote.database.protocol.protocol import ProtocolFile
from allies.utils import get_protocols

def speakers_to_annotation(speakers, uri = None, modality = 'speaker'):
    annotation = Annotation(uri, modality)
    for idx in range(speakers.speaker.size):
        start=speakers.start_time[idx]
        stop=speakers.end_time[idx]
        segment = Segment(start, stop)
        annotation[segment, idx] = speakers.speaker[idx]
    return annotation

def load_protocol(train_generator,dev_generator) -> SpeakerDiarizationProtocol:
        """Given a ProtocolFile generator, instantiate a SpeakerDiarizationProtocol
        which yields the file in the relevant subset_iter
        depending on the uris returned from get_protocols()

        Parameters
        ----------
        train_generator,dev_generator : generator
            yields ProtocolFile
            generator are duplicated this way we can iterate on both of them
            and filter out files

        Returns
        -------
        protocol : SpeakerDiarizationProtocol instance
        """
        print("loading data in protocol")
        protocols=get_protocols()
        class AlliesProtocol(SpeakerDiarizationProtocol):

            def trn_iter(self):
                for current_file in train_generator:
                    if current_file['uri'] in protocols['train']:
                        yield current_file

            def dev_iter(self):
                for current_file in dev_generator:
                    if current_file['uri'] in protocols['development']:
                        yield current_file

            def tst_iter(self):
                raise ValueError(
                    "Initial training protocol doesn't have a test set, "
                    "you should use `diar_lifelong`"
                    )

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
        uem = UEM(data["uem"],file_id)
        features = data['features'].value
        if len(features.shape) != 2:
            print(features.shape)
            msg = (
                f'Precomputed waveform should be provided as a '
                f'(n_samples, n_channels) `np.ndarray`.'
            )
            print(msg)
            raise ValueError(msg)
        annotated = uem.to_timeline()
        annotation = speakers_to_annotation(speakers, file_id).crop(annotated)
        current_file = {
            'uri' : file_id,
            'annotation' : annotation,
            'annotated' : annotated,
            'waveform' : features
        }
        yield ProtocolFile(current_file)


class UEM:
    """
    A wrapper of an ALLIES UEM dict.
    Provides handy methods to work with pyannote.
    """

    def __init__(self, uem, uri = None):
        """Parameters:
        -----------
        uem: dict or uemranges
          - 'start_time': list[float], start times of speech in seconds
          - 'end_time': list[float], end times of speech in seconds
          uem is converted to dict if it's uemranges
        uri : str, optional
            Uniform Resource Identifier of the Timeline.
            Defaults to None.
        """
        self.uem = uem
        self.uri = uri
        if not isinstance(self.uem,dict):
            self.uem = {
                'start_time':self.uem.start_time,
                'end_time':self.uem.end_time
            }

    def to_timeline(self):
        """Converts Allies-UEM to pyannote Timeline

        Parameters:
        -----------
        self.uem: dict
          - 'start_time': list[float], start times of speech in seconds
          - 'end_time': list[float], end times of speech in seconds


        Returns:
        --------
        timeline, see pyannote.core.Timeline
        """
        segments = [Segment(start, end) for start, end in zip(self.uem['start_time'], self.uem['end_time'])]
        return Timeline(segments, self.uri)

    def to_annotation(self):
        """Converts Allies-UEM to pyannote Annotation

        Parameters:
        -----------
        self.uem: dict of
          - 'start_time': list[float], start times of speech in seconds
          - 'end_time': list[float], end times of speech in seconds

        Returns:
        --------
        annotation with string labels, see pyannote.core.Annotation
        """
        return self.to_timeline().to_annotation(generator='string')


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
