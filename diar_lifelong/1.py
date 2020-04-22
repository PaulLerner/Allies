#!/usr/bin/env python
from allies.utils import get_params, hypothesis_to_unk
from allies.serializers import DummySerializer
from allies.convert import UEM, AlliesAnnotation, id_to_file, Time
from allies.distances import get_thresholds, get_farthest, find_closest_to
from pyannote.audio.pipeline import SpeakerDiarization, SpeakerIdentification, update_references, get_references
import numpy as np
from pyannote.audio.features.wrapper import Wrapper
from pyannote.database import get_protocol

class Algorithm:
    """
    Lifelong diarization module
    """

    def __init__(self):
        self.database = "ALLIES"
        self.protocol = "ALLIES.SpeakerDiarization.Custom"
        self.id_to_file = id_to_file()
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
                                              scd_scores = '@scd',
                                              embedding = '@emb',
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
        #load references from data_loaders the first time we call process
        # also compute distance thresholds
        if self.identification is None:
            references = get_references(self.protocol,
                                        self.model['emb'],
                                        subsets={'train', 'development'})
            self.thresholds = get_thresholds(references, self.parameters['pipeline']['params'].get("metric","cosine"))
            self.identification = SpeakerIdentification(references,
                                                        sad_scores = 'oracle',
                                                        scd_scores = '@scd',
                                                        embedding = '@emb',
                                                        metric = self.parameters['pipeline']['params'].get("metric","cosine"),
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
                'annotated': uem.to_timeline(),
                #pyannote needs the real file name to use precomputed features
                'uri': self.id_to_file[uri],
                'database': self.database
               }
        #compute SCD scores and embeddings
        scd, emb = Wrapper(self.model['scd']), Wrapper(self.model['emb'])
        file['scd'], file['emb'] = scd(file), emb(file)
        human_assisted_learning = supervision in {"active", "interactive"}

        if not human_assisted_learning:
            #TODO: unsupervised adaptation
            pass
        # 1. assign each segment to the closest reference if close enough
        # else tag with negative label
        hypothesis = self.identification(file, use_threshold = True)
        alliesAnnotation = AlliesAnnotation(hypothesis).to_hypothesis()
        # cluster < 0 (unk)
        unknown = hypothesis_to_unk(hypothesis)
        unknown = self.diarization.speech_turn_clustering(file, unknown) if unknown else unknown

        #keep track of segments which come from same (resp. different) speakers
        positives, negatives = [], []
        # If human assisted learning mode is on (active or interactive learning)
        # and we still have some segments we're unsure about
        while human_assisted_learning and unknown:
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
               "time_2": np.float32(1.5)
              }
            # FFQS-explore unknown based on centroids and thresholds
            if supervision == "active":
                # The system can send a question to the human in the loop
                # by using an object of type request
                # The request is the question asked to the system
                # 1. find segment farthest from all existing clusters
                farthest_speaker, farthest_segment, farthest_embedding = get_farthest(file,
                                                                                      unknown,
                                                                                      '@emb',
                                                                                      metric=self.parameters['pipeline']['params'].get("metric","cosine"))
                #del farthest_segment from unknown so we don't query it again
                del unknown[farthest_segment]

                #time_1 is the middle time of the farthest_segment
                time_1 = farthest_segment.middle

                #2. find segment closest to the farthest_segment
                closest_speaker, closest_segment = find_closest_to(farthest_segment,
                                                                  farthest_embedding,
                                                                  file,
                                                                  hypothesis,
                                                                  '@emb',
                                                                  metric=self.parameters['pipeline']['params'].get("metric","cosine"))
                time_2 = closest_segment.middle

                request = {
                   "request_type": "same",
                   "time_1": np.float32(time_1),
                   "time_2": np.float32(time_2)
                  }

            # Send the request to the user and wait for the answer
            message_to_user = {
                "file_id": uri,  # ID of the file the question is related to
                "hypothesis": alliesAnnotation,  # The current hypothesis
                "system_request": request  # the question for the human in the loop
            }
            human_assisted_learning, user_answer = loop_channel.validate(message_to_user)
            response_type = user_answer["response_type"]
            if response_type == 'stop':
                #user is tired of us
                break
            elif response_type == 'boundary':
                #TODO: update SCD model or pipeline
                pass
            elif response_type == 'same':
                # Time to segment / hypothesis labels
                time_1, time_2 = Time(user_answer['time_1']), Time(user_answer['time_2'])
                same = user_answer["answer"]["value"]
                if supervision == "active":
                    if time_1.in_segment(farthest_segment):
                        s1, l1 = farthest_segment, farthest_speaker
                    else:
                        print(f'expected time_1 ({time_1}) to be in '
                              f'farthest_segment {farthest_segment} as queried')
                        s1, t1, l1 = time_1.find_label(hypothesis)
                    if time_2.in_segment(closest_segment):
                        s2, l2 = closest_segment, closest_speaker
                    else:
                        print(f'expected time_2 ({time_2}) to be in '
                              f'closest_segment {closest_segment} as queried')
                        s2, t2, l2 = time_2.find_label(hypothesis)
                else:
                    s1, t1, l1 = time_1.find_label(hypothesis)
                    s2, t2, l2 = time_2.find_label(hypothesis)
                # make use of user answer
                # TODO: propagate to close segments ?
                if same:
                    positives.append((s1,s2))
                    if supervision == "active":
                        #system initiative
                        # relabel queried segment
                        del hypothesis[s1]
                        hypothesis[s1] = l2
                    else:
                        #user initiative
                        #prefer already existing references
                        if l2 in self.identification.references:
                            del hypothesis[s1]
                            hypothesis[s1] = l2
                        elif l1 in self.identification.references:
                            del hypothesis[s2]
                            hypothesis[s2] = l1
                        else:
                            #merge the two clusters
                            hypothesis.rename_labels(mapping={l1:l2}, copy=False)
                            unknown.rename_labels(mapping={l1:l2}, copy=False)
                else:
                    negatives.append((s1,s2))
            else:
                print(f'got an unexpected response type from user: {response_type}')

            # TODO: fine-tune embedding model given positives, negatives
            alliesAnnotation = AlliesAnnotation(hypothesis).to_hypothesis()

            # hypothesis = self.identification(file, use_threshold = True)
            #
            # # cluster < 0 (unk)
            # # BEWARE before un-commenting :
            # # the same segments can be queried again and again if you update unknown
            # unknown = hypothesis_to_unk(hypothesis)
            # unknown = self.diarization.speech_turn_clustering(file, unknown)


        # update references with the new clusters
        mapping = {label: f'{uri}#{label}' for label in unknown.labels()}
        unknown.rename_labels(mapping=mapping,copy=False)
        update_references(file, unknown, '@emb', self.identification.references)
        #make final hypothesis with the new references
        hypothesis = self.identification(file, use_threshold = False)
        alliesAnnotation = AlliesAnnotation(hypothesis).to_hypothesis()

        # End of human assisted learning
        # Send the current hypothesis
        outputs["adapted_speakers"].write(alliesAnnotation)

        # FIXME what is this ?? (from anthony baseline)
        if not inputs.hasMoreData():
            pass

        # always return True, it signals BEAT to continue processing
        return True

if __name__ == '__main__':
    #DEBUG : instantiate Algorithm
    algorithm=Algorithm()
    print('instantiated algorithm')
    references = get_references(algorithm.protocol,
                algorithm.model['emb'],
                subsets={'train', 'development'})
    algorithm.thresholds = get_thresholds(references, algorithm.parameters['pipeline']['params'].get("metric","cosine"))
    algorithm.identification = SpeakerIdentification(references,
                        sad_scores = 'oracle',
                        scd_scores = '@scd',
                        embedding = '@emb',
                        metric = algorithm.parameters['pipeline']['params'].get("metric","cosine"),
                        evaluation_only = algorithm.parameters['pipeline']['params'].get("evaluation_only",False))
    print('init identification pipeline ok')
    params = {
        'closest_assignment' : {'threshold':algorithm.thresholds['close']},
        'speech_turn_segmentation': algorithm.parameters['params']['speech_turn_segmentation']
    }
    algorithm.identification.instantiate(params)
    print('instantiate pipeline ok')
