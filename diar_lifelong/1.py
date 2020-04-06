#!/usr/bin/env python
import numpy as np
import pickle
import struct
from pyannote.audio.utils.signal import Peak
from pyannote.audio.features import Pretrained
from allies.convert import uem_to_timeline

def deserialize(model_loader):
    """
    Deserialize a trained model given its loader
    :param model_loader: BEAT loader for the model
    :return: a dict of
      - 'scd': `Pretrained` instance for the SCD model
      - 'emb': `Pretrained` instance for the speaker embedding model
      - 'clustering': still to be defined
    """
    tmp_m = model_loader[0][0]['model'].value
    pkl_after = struct.pack('{}B'.format(len(tmp_m)), *list(tmp_m))
    serialized_model = pickle.loads(pkl_after)
    return {
        # TODO Passing paths for now to debug
        # This will surely be model weights in 'production'
        'scd': Pretrained(validate_dir=serialized_model['scd']),
        'emb': Pretrained(validate_dir=serialized_model['emb']),
        # TODO Still don't know how this model will be formatted
        'clustering': serialized_model['clustering']
    }


class Algorithm:
    """
    Lifelong diarization module
    """

    def __init__(self):
        self.model = None
        self.sample_rate = 16000

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
          - 'speakers': list[str] list of predicted speaker IDs
          - 'start_time': list[float], start times of speakers in seconds
          - 'end_time': list[float], end times of speakers in seconds
        :param loop_channel: ??
        :return: always True to keep processing
        """
        # Load model if it's the first time the module runs
        if self.model is None:
            self.model = deserialize(data_loaders.loaderOf("model"))

        # File input
        wave = inputs['features'].data.value
        uem = inputs['processor_uem'].data
        uri = inputs['processor_file_info'].get('file_id')

        # Build timeline from UEM SAD annotations
        speech = uem_to_timeline(uem, uri)

        # FIXME the following lines only predict speakers, no model updates yet
        # FIXME replace with SpeakerDiarization pipeline if possible

        # Calculate SCD scores and speaker turns
        scd_scores = self.model['scd'].get_features(wave, self.sample_rate)
        # TODO tune these values during training/lifelong stage
        peak = Peak(alpha=0.10, min_duration=0.10, log_scale=True)
        partition = peak.apply(scd_scores, dimension=1)

        # Intersection between speech and speaker turns
        speech_turns = partition.crop(speech)

        # Calculate speaker embeddings
        embeddings = self.model['emb'].get_features(wave, self.sample_rate)
        embs, start_times, end_times = [], [], []
        for segment in speech_turns:
            # "strict" because we don't want to look outside speech turns
            emb = embeddings.crop(segment, mode='strict')
            # Average embeddings for this segment
            embs.append(np.mean(emb, axis=0))
            start_times.append(segment.start)
            end_times.append(segment.end)
        embs = np.vstack(embs)

        # Get speaker IDs from clustering model
        # FIXME just a placeholder for the actual clustering code
        speakers = [self.model['clustering'].predict(emb) for emb in embs]

        # Write predictions
        hypothesis = {"speaker": speakers,
                      "start_time": start_times,
                      "end_time": end_times}
        outputs["adapted_speakers"].write(hypothesis)

        return True
