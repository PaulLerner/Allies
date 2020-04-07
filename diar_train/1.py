from allies.serializers import DummySerializer


class Algorithm:
    """
    Diarization training module.
    We return paths to pretrained models for the time being,
    as it's easier to debug
    """
    def __init__(self):
        self.serializer = DummySerializer()

    def process(self, data_loaders, outputs):
        """
        Use the training data provided through the dataloader in order to 
        train the acoustic model and return it

        :param data_loader: input parameters that is used to access all incoming data
        :param outputs: parameter to send the model to the next block
        """
        # The laoder is the interface used to access inputs of this algorithmic block
        # loader = data_loaders.loaderOf("features")

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


