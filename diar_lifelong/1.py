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
        self.pipeline = None
        self.hyperparameters = {
        #AMI hyperparameters from pyannote tutorial
        #TODO update with ALLIES hyperparameters
        "pipeline": {
            "min_duration": 3.306092065580709,
            "speech_turn_assignment": {
                "closest_assignment": {
                    "threshold": 0.8401481964056187
                }
            },
            "speech_turn_clustering": {
                "clustering": {
                    "damping": 0.6066098204003955,
                    "preference": -2.9717704925136976
                }
            },
            "speech_turn_segmentation": {
                "speaker_change_detection": {
                    "alpha": 0.11115647156273972,
                    "min_duration": 0.5283486365753665
                },
                # we use oracle SAD from UEM
                "speech_activity_detection": {}
                }
            }
        }


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

        # Load pipeline if it's the first time the module runs
        if self.pipeline is None:
            # Build diarization pipeline
            self.pipeline = SpeakerDiarization(sad_scores='oracle',
                                              scd_scores=self.model['scd'],
                                              embedding=self.model['emb'])
            #instantiate pipeline with hyperparameters computed offline
            self.pipeline.instantiate(self.hyperparameters['pipeline'])

        # ALLIES lifelong step inputs
        wave = inputs['features'].data.value
        uri = inputs['processor_file_info'].get('file_id')
        uem = UEM(inputs['processor_uem'].data,uri)

        # Build input to pyannote pipeline
        file = {'waveform': wave,
                'annotation': uem.to_annotation()}

        # FIXME only predicting speakers, no model updates yet
        hypothesis = AlliesAnnotation(self.pipeline(file)).to_hypothesis()

        # Write output
        outputs["adapted_speakers"].write(hypothesis)

        # Always return True
        return True
