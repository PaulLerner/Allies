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
            # check fake protocol durations
            # TODO update with real protocol.stats
            annotated_total, annotation_total = 0.0,0.0
            for uri, current_file in self.protocol.items() :
                annotated = current_file['annotated']
                annotated_duration = annotated.duration()
                annotation = current_file['annotation']
                annotation_duration = annotation.get_timeline().duration()
                print(f"{uri}, annotation: {annotation_duration:.2f}, annotated: {annotated_duration:.2f}")
                annotated_total+=annotated_duration
                annotation_total+=annotation_duration
            print(f'\n\nTOTAL:  annotation: {annotation_total:.2f}, annotated: {annotated_total:.2f}')

        # TODO add missing models
        scd = "don't have it yet"
        emb = "/vol/work2/coria/allies/AAM/train/" + \
              "ALLIES.SpeakerDiarization.Debug.train/" + \
              "validate_diarization_fscore/" + \
              "ALLIES.SpeakerDiarization.Official.development"
        clustering = "don't have it yet"

        outputs["model"].write({"value": serialize(scd, emb, clustering)})
        return True
