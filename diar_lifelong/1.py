#!/usr/bin/env python
from allies.utils import get_params, label_generator
from allies.serializers import DummySerializer
from allies.convert import UEM, AlliesAnnotation
from allies.distances import get_thresholds
from pyannote.audio.pipeline import SpeakerDiarization, SpeakerIdentification, update_references, get_references
import numpy as np

class Algorithm:
    """
    Lifelong diarization module
    """

    def __init__(self):

        #load parameters from config yml file
        self.parameters = get_params()
        #use local, evolving model if provided
        #else relies on precomputed scores
        self.model = {
            'scd': self.parameters['model'].get('scd',self.parameters['pipeline']['params']['scd_scores']),
            'emb': self.parameters['model'].get('emb',self.parameters['pipeline']['params']['embedding'])
        }
        #load pipeline from parameters
        self.diarization = SpeakerDiarization(sad_scores = 'oracle',
                                              scd_scores = self.model['scd'],
                                              embedding = self.model['emb'],
                                              metric = self.parameters['pipeline']['params'].get("metric","cosine"),
                                              method = self.parameters['pipeline']['params'].get("method","pool"),
                                              evaluation_only = self.parameters['pipeline']['params'].get("evaluation_only",False),
                                              purity = self.parameters['pipeline']['params'].get("purity",None))

        #instantiate pipeline from parameters
        self.diarization.instantiate(self.parameters['params'])

        # identification pipeline is instantiated the first time process is called
        self.identification = None

    def process(self, inputs, data_loaders, outputs, loop_channel):
        """
        Execute one step of the lifelong process:
          - Load input data
          - Make a prediction (hypothesis)
          - Update model (unsupervised/interactive/active)
        :param inputs: a dict object of
          - 'features': audio waveform, shape (seq_len, 1)
          - 'processor_uem': uemranges
            - 'start_time': list[float], start times of speech in seconds
            - 'end_time': list[float], end times of speech in seconds
          - 'processor_file_info': file_info_sd
            - 'file_id': str
            - 'timestamp': str
            - 'supervision': str
        :param data_loaders: BEAT data loader accessor
        :param outputs: where we need to write an entry 'adapted_speakers' with
          - 'speaker': list[str] list of predicted speaker IDs
          - 'start_time': list[float], start times of speakers in seconds
          - 'end_time': list[float], end times of speakers in seconds
        :param loop_channel: ??
            Used to interact with the user
        :return: always True to keep processing
        """
        # ALLIES lifelong step inputs
        wave = inputs['features'].data.value
        file_info = inputs["processor_file_info"].data
        uri = file_info.file_id
        supervision = file_info.supervision
        time_stamp = file_info.time_stamp
        uem = inputs['processor_uem'].data
        uem = UEM(uem,uri)
        annotated=uem.to_annotation()

        #load references from data_loaders the first time we call process
        # also compute distance thresholds
        if self.identification is None:
            references = get_references("ALLIES.SpeakerDiarization.Custom",
                                                         self.model['emb'],
                                                         subsets={'train', 'development'})
            self.thresholds = get_thresholds(references, self.parameters['pipeline']['params'].get("metric","cosine"))
            self.identification = SpeakerIdentification(references,
                                                        sad_scores = 'oracle',
                                                        scd_scores = self.model['scd'],
                                                        embedding = self.model['emb'],
                                                        metric = self.parameters['pipeline']['params'].get("metric","cosine"),
                                                        method = self.parameters['pipeline']['params'].get("method","pool"),
                                                        evaluation_only = self.parameters['pipeline']['params'].get("evaluation_only",False))
            params = {
                'closest_assignment' : {'threshold':self.thresholds['close']},
                'speech_turn_segmentation': self.parameters['params']['speech_turn_segmentation']
            }
            self.identification.instantiate(params)
        else:
            #TODO: update SpeakerIdentification thresholds if embedding model is evolving
            pass

        # Build input to pyannote pipeline
        file = {'waveform': wave,
                'annotation': uem.to_annotation(),
                'annotated': uem.to_timeline()
               }

        human_assisted_learning = supervision in {"active", "interactive"}

        if not human_assisted_learning:
            #TODO: unsupervised adaptation
            pass
        # FIXME only predicting speakers, no model updates yet
        # 1. assign each segment to the closest reference if close enough
        # else tag with '?'
        # FIXME : use SpeakerIdentification pipeline instead of Diarization
        hypothesis = self.identification(file, use_thresholds = True)

        # cluster '?' (unk)
        unknown = hypothesis.empty()
        for segment, track, label in hypothesis.itertracks(yield_label=True):
            if label == '?':
                unknown[segment, track] = label
        unknown = self.diarization.speech_turn_clustering(file, unknown)
        unknown.rename_labels(generator=label_generator(),copy=False)
        # update references with the new clusters
        update_references(file, unknown, self.model['emb'], self.identification.references)

        hypothesis = self.identification(file, use_thresholds = False)
        alliesAnnotation = AlliesAnnotation(hypothesis)
        alliesAnnotation = alliesAnnotation.to_hypothesis()

        # If human assisted learning mode is on (active or interactive learning)
        while human_assisted_learning:
            # Create an empty request that is used to initiate interactive learning
            # For the case of active learning, this request is overwritten by your system itself
            # request_type can be either "same" or "boundary"
            # if "same", the question asked to the user is: Is the same speaker speaking at time time_1 and time_2
            #            the cost of this question is 6s / total_file_duration
            # if "boundary" the question asked to the user is: What are the boundaries of the segment including time_1
            #            note that the data time_2 is not used in this request
            #            the cost of this question is (|time_2 - time_1| + 6s) / total_file_duration
            request = {
               "request_type": "same",
               "time_1": np.float32(0.5),
               "time_2": np.float32(1.5),
              }

            # FFQS-explore unknown based on centroids and thresholds
            if supervision == "active":
                pass
                # The system can send a question to the human in the loop
                # by using an object of type request
                # The request is the question asked to the system
                #TODO
                #request = generate_system_request_to_user(self.model, self.global_diar, current_s4d_segmentation)

            # Send the request to the user and wait for the answer
            message_to_user = {
                "file_id": uri,  # ID of the file the question is related to
                "hypothesis": alliesAnnotation,  # The current hypothesis
                "system_request": request,  # the question fot the human in the loop
            }
            human_assisted_learning, user_answer = loop_channel.validate(message_to_user)

            # TODO
            # Take into account the user answer to generate a new hypothesis
            # and possibly update the model
            hypothesis = self.identification(file, use_thresholds = False)

            # X. convert hypothesis to AlliesAnnotation
            alliesAnnotation = AlliesAnnotation(hypothesis)
            alliesAnnotation = alliesAnnotation.to_hypothesis()

        # End of human assisted learning
        # Send the current hypothesis
        outputs["adapted_speakers"].write(alliesAnnotation)

        if not inputs.hasMoreData():
            pass
        # always return True, it signals BEAT to continue processing
        return True
