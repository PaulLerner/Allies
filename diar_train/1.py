from allies.serializers import DummySerializer
import numpy as np
import struct
import pickle
from allies.convert import yield_train, load_protocol
from allies.utils import print_stats
from itertools import tee

class Algorithm:
    """
    Diarization training module.
    We return paths to pretrained models for the time being,
    as it's easier to debug
    """

    def __init__(self):
        self.serializer = DummySerializer()
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
            #file_generator_ should not be used directly
            #use train_generator or dev_generator instead
            file_generator_ = yield_train(data_loaders)
            train_generator,dev_generator = tee(file_generator_)
            self.protocol = load_protocol(train_generator,dev_generator)
            print('getting stats from protocol train subset...')
            print_stats(self.protocol.stats('train'))
            print('getting stats from protocol development subset...')
            print_stats(self.protocol.stats('development'))

        scd = "/vol/work2/coria/allies/scd/train/" + \
              "ALLIES.SpeakerDiarization.Official.train/" + \
              "validate_segmentation_fscore/" + \
              "ALLIES.SpeakerDiarization.Official.development"

        emb = "/vol/work2/coria/allies/AAM/train/" + \
              "ALLIES.SpeakerDiarization.Debug.train/" + \
              "validate_diarization_fscore/" + \
              "ALLIES.SpeakerDiarization.Official.development"

        model = {'scd': scd, 'emb': emb}

        outputs["model"].write({"value": self.serializer.serialize(model)})
        return True
