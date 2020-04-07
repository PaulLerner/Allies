import numpy as np
import struct
import pickle
from allies.convert import load_train

def serialize(scd, emb, clustering):
    """
    Serialize models to an array of uint8
    """
    model = dict()
    # TODO serialize model weights?
    model["scd"] = scd
    model["emb"] = emb
    model["clustering"] = clustering
    pkl = pickle.dumps(model)
    u8 = np.array(struct.unpack("{}B".format(len(pkl)), pkl), dtype=np.uint8)
    return u8

class Algorithm:
    """
    Diarization training module.
    We return paths to pretrained models for the time being,
    as it's easier to debug
    """

    def __init__(self):
        self.model = None
        self.sample_rate = 16000
        self.protocol = None

    def process(self, data_loaders, outputs):
        """
        Use the training data provided through the dataloader in order to
        train the acoustic model and return it

        :param data_loader: input parameters that is used to access all incoming data
        :param outputs: parameter to send the model to the next block
        """
        # The laoder is the interface used to access inputs of this algorithmic block
        # loader = data_loaders.loaderOf("features")

        # Load model if it's the first time the module runs
        if self.model is None:
            pass
            #self.model = TODO

        # Load protocol if it's the first time the module runs
        if self.protocol is None:
            self.protocol = load_train(data_loaders)

        # File input
        wave = inputs['features'].data.value
        uem = inputs['processor_uem'].data
        uri = inputs['processor_file_info'].get('file_id')

        # TODO add missing models
        scd = "don't have it yet"
        emb = "/vol/work2/coria/allies/AAM/train/" + \
              "ALLIES.SpeakerDiarization.Debug.train/" + \
              "validate_diarization_fscore/" + \
              "ALLIES.SpeakerDiarization.Official.development"
        clustering = "don't have it yet"

        outputs["model"].write({"value": serialize(scd, emb, clustering)})
        return True
