import numpy as np
import pickle
import struct


class Serializer:

    def serialize(self, model):
        """
        Serialize a model dict to a string of uint8 bytes
        :param model: a dict with model components
        :return: a np.array of uint8
        """
        raise NotImplementedError

    def deserialize(self, model):
        """
        Deserialize a model from a string of uint8 bytes
        :param model: a np.array of uint8
        :return: a dict with model components
        """
        raise NotImplementedError


class DummySerializer(Serializer):
    """
    Dummy serializer for debug purposes.
    We assume inputs in the model dict are just paths to pretrained models
    """

    def serialize(self, model):
        pkl = pickle.dumps(model)
        u8 = np.array(struct.unpack("{}B".format(len(pkl)), pkl), dtype=np.uint8)
        return u8

    def deserialize(self, model):
        pkl_after = struct.pack('{}B'.format(len(model)), *list(model))
        serialized_model = pickle.loads(pkl_after)
        return serialized_model