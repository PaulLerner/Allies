import numpy as np

class Algorithm:
    """
    Feature extraction module.
    The architecture learns feature extraction jointly with the task,
    so we just return the waveform as a 2D array
    """
    
    def process(self, inputs, dataloader, outputs):
        """
        Doesn't really do anything, just output the waveform as a 2D array
        :param inputs: dict of
          - 'speech' waveform as a 1D array
        :param dataloader: BEAT data loader accessor
        :param outputs: where we need to write an entry
          'features' with the waveform having shape (seq_len, 1)
        :return:
        """
        speech = np.expand_dims(inputs["speech"].data.value, axis=1)
        outputs["features"].write({"value": speech})
        return True
 
