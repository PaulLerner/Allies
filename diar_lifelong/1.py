#!/usr/bin/env python
from pyannote.audio.features import Pretrained
from allies.serializers import DummySerializer
from allies.convert import UEM, AlliesAnnotation
from pyannote.audio.pipeline import SpeakerDiarization


class Algorithm:
    """
    Lifelong diarization module
    """

    def __init__(self):
        self.model = None
        self.serializer = DummySerializer()

    def process(self, inputs, data_loaders, outputs, loop_channel):
        """
        Execute one step of the lifelong process:
          - Load input data
          - Make a prediction (hypothesis)
          - Update model (unsupervised/interactive/active)
        :param inputs: a dict object of
          - 'features': audio waveform, shape (seq_len, 1)
          - 'processor_uem': dict of
            - 'start_time': list[float], start times of speech in seconds
            - 'end_time': list[float], end times of speech in seconds
          - 'processor_file_info': dict of
            - 'file_id': str
            - 'timestamp': str
            - 'supervision': str
        :param data_loaders: BEAT data loader accessor
        :param outputs: where we need to write an entry 'adapted_speakers' with
          - 'speaker': list[str] list of predicted speaker IDs
          - 'start_time': list[float], start times of speakers in seconds
          - 'end_time': list[float], end times of speakers in seconds
        :param loop_channel: ??
        :return: always True to keep processing
        """
        # Load model if it's the first time the module runs
        if self.model is None:
            model = data_loaders.loaderOf("model")[0][0]['model'].value
            self.model = self.serializer.deserialize(model)

        # ALLIES lifelong step inputs
        wave = inputs['features'].data.value
        uri = inputs['processor_file_info'].get('file_id')
        uem = UEM(inputs['processor_uem'].data,uri)

        # Build input to pyannote pipeline
        file = {'waveform': wave,
                'annotation': uem.to_annotation()}

        # Build diarization pipeline
        pipeline = SpeakerDiarization(sad_scores='oracle',
                                      scd_scores=self.model['scd'],
                                      embedding=self.model['emb'])

        # FIXME only predicting speakers, no model updates yet
        hypothesis = AlliesAnnotation(pipeline(file)).to_hypothesis()

        # Write output
        outputs["adapted_speakers"].write(hypothesis)

        # Always return True
        return True
