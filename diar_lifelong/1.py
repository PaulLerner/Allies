#!/usr/bin/env python
from datetime import datetime
import numpy as np
from pathlib import Path

from pyannote.audio.pipeline import SpeakerDiarization, KNearestSpeakers, \
    update_references, get_references
from pyannote.audio.features.wrapper import Wrapper
from pyannote.database import get_protocol

import allies
from allies.utils import get_params, hypothesis_to_unk, relabel_unknown
from allies.serializers import DummySerializer
from allies.convert import UEM, AlliesAnnotation, id_to_file, Time
from allies.distances import get_thresholds, get_farthest, find_closest_to, get_centroids

TIMESTAMP = datetime.today().strftime('%Y%m%d-%H%M%S')
APPLY_DIR = Path(allies.__file__).parent / 'apply'
APPLY_DIR.mkdir(exist_ok=True)
SAVE_TO = APPLY_DIR / f'{TIMESTAMP}.rttm'
print(f'Hypotheses will be saved to {SAVE_TO}')


class Algorithm:
    """
    Lifelong diarization module
    """

    def __init__(self):
        self.database = "ALLIES"
        self.protocol = "ALLIES.SpeakerDiarization.Custom"
        self.id_to_file = id_to_file()
        # load parameters from config yml file
        self.parameters = get_params()
        # use local, evolving model if provided
        # else relies on precomputed scores
        self.model = {
            'scd': self.parameters.get('model',{}).get('scd',self.parameters['pipeline']['params']['scd_scores']),
            'emb': self.parameters.get('model',{}).get('emb',self.parameters['pipeline']['params']['embedding'])
        }
        # load diarization pipeline from parameters
        self.metric = self.parameters['pipeline']['params'].get("metric", "cosine")
        self.clustering_method = self.parameters['pipeline']['params'].get("method", "affinity_propagation")
        self.evaluation_only = self.parameters['pipeline']['params'].get("evaluation_only", True)
        self.diarization = SpeakerDiarization(sad_scores='oracle',
                                              scd_scores='@scd',
                                              embedding='@emb',
                                              metric=self.metric,
                                              method=self.clustering_method,
                                              evaluation_only=self.evaluation_only,
                                              purity=self.parameters['pipeline']['params'].get("purity"))

        # instantiate pipeline from parameters
        self.diarization.instantiate(self.parameters['params'])

        args = self.parameters.get('identification',{})
        # load identification pipeline from parameters
        references = get_references(self.protocol,
                                    self.model['emb'],
                                    args.get('subsets',{'train', 'development'}),
                                    label_min_duration=args.get('label_min_duration', 0.0))
        # use thresholds tuned on dev set
        # self.thresholds = get_thresholds(references, self.metric)
        self.identification = KNearestSpeakers(references,
                                               sad_scores='oracle',
                                               scd_scores='@scd',
                                               embedding='@emb',
                                               metric=self.metric,
                                               evaluation_only=self.evaluation_only,
                                               confusion=False,
                                               weigh=args.get('weigh', True))
        params = {
            'classifier': args['classifier'],
            'speech_turn_segmentation': self.parameters['params']['speech_turn_segmentation']
        }
        self.identification.instantiate(params)

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
        uem = UEM(uem, uri)

        # Build input to pyannote pipeline
        file = {'waveform': wave,
                'annotation': uem.to_annotation(),
                'annotated': uem.to_timeline(),
                # pyannote needs the real file name to use precomputed features
                'uri': self.id_to_file[uri],
                'database': self.database
                }
        # compute SCD scores and embeddings
        scd, emb = Wrapper(self.model['scd']), Wrapper(self.model['emb'])
        file['scd'], file['emb'] = scd(file), emb(file)
        human_assisted_learning = supervision in {"active", "interactive"}
        active = supervision == "active"
        if not human_assisted_learning:
            # TODO: unsupervised adaptation
            pass
        # 1. assign each segment to the closest reference if close enough
        # else tag with negative `int` label
        hypothesis = self.identification(file, use_threshold=True)

        # mutually cannot-link identified speakers
        cannot_link = mutual_cl(hypothesis)

        # cluster speakers (taking into account identified ones)
        hypothesis = self.diarization.speech_turn_clustering(file, hypothesis)
        alliesAnnotation = AlliesAnnotation(hypothesis).to_hypothesis()

        # keep track of segments which come from same (resp. different) speakers
        positives, negatives = [], []

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
                "time_2": np.float32(1.5)
            }
            # FFQS-explore hypothesis based on centroids and thresholds
            if active:
                # The system can send a question to the human in the loop
                # by using an object of type request
                # The request is the question asked to the system

                # find segment farthest from all existing clusters in the current hypothesis
                query_speaker, farthest, centroid = get_farthest(file,
                                                                 hypothesis,
                                                                 '@emb',
                                                                 metric=self.metric)
                time_1, time_2 = farthest.middle, centroid.middle
                # is this farthest segment in the right cluster ?
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
            print('got user_answer :', user_answer)
            response_type = user_answer.response_type
            if response_type == 'stop':
                # user is tired of us
                break
            elif response_type == 'boundary':
                # TODO: update SCD model or pipeline
                pass
            elif response_type == 'same':
                # Time to segment / hypothesis labels
                time_1, time_2 = Time(user_answer.time_1), Time(user_answer.time_2)
                same = user_answer.answer.value
                if active:
                    s1, l1 = farthest, query_speaker
                    s2, l2 = centroid, query_speaker
                else:
                    s1, t1, l1 = time_1.find_label(hypothesis)
                    s2, t2, l2 = time_2.find_label(hypothesis)
                print(time_1, s1, l1)
                print(time_2, s2, l2)
                # make use of user answer
                # TODO: propagate to close segments ?
                if same:
                    positives.append((s1, s2))
                    if active:
                        # system initiative -> relabel queried segment
                        # 1. convert centroid label to string so both segments are merged
                        if not isinstance(l2, str):
                            l2 = str(l2)
                            del hypothesis[s2]
                            hypothesis[s2] = l2
                        print(f'relabel {s1}: {l1} <- {l2}')
                        # 2. relabel queried segment
                        del hypothesis[s1]
                        hypothesis[s1] = l2
                    # user initiative -> prefer already existing references
                    elif l2 in self.identification.references:
                        print(f'relabel {s1}: {l1} <- {l2}')
                        del hypothesis[s1]
                        hypothesis[s1] = l2
                    elif l1 in self.identification.references:
                        print(f'relabel {s2}: {l2} <- {l1}')
                        del hypothesis[s2]
                        hypothesis[s2] = l1
                    else:
                        # create a new `str` label so both segments are merged
                        new_label = f'{l1}@{l2}'
                        print(f'relabel {s1}: {l1} <- {new_label}')
                        print(f'relabel {s2}: {l2} <- {new_label}')
                        del hypothesis[s1]
                        del hypothesis[s2]
                        hypothesis[s1] = new_label
                        hypothesis[s2] = new_label
                        # rename str(l1) -> new_label in case it already exists to propagate
                        # annotations. Note that we enforce `str` so we don't relabel
                        # other segments in the hypothetic un-indentified cluster l1
                        hypothesis.rename_labels(mapping={str(l1):new_label}, copy=False)
                        hypothesis.rename_labels(mapping={str(l2):new_label}, copy=False)

                else:
                    negatives.append((s1, s2))
                    print(f'cannot-link {s1} : {s2}')
                    # update cannot-links
                    cannot_link.setdefault(s1, [])
                    cannot_link[s1].append(s2)
            else:
                print(f'got an unexpected response type from user: {response_type}')

            # TODO: fine-tune embedding model given positives, negatives
            # update hypothesis with constraints
            # 1. relabel unknown segments with a unique label so they're not merged together
            hypothesis = relabel_unknown(hypothesis)
            # 2. apply the actual pipeline
            hypothesis = self.diarization.speech_turn_clustering(file, hypothesis,
                                                                 cannot_link = cannot_link)
            alliesAnnotation = AlliesAnnotation(hypothesis).to_hypothesis()

        # write hypothesis locally to a time-stamped file
        with open(SAVE_TO, 'a') as file:
            hypothesis.write_rttm(file)

        # End of human assisted learning
        # Send the current hypothesis
        outputs["adapted_speakers"].write(alliesAnnotation)

        # update references with new clusters and newly identified speakers
        # relabel new clusters so they don't mix with previous ones
        mapping = {label: f'{uri}#{label}' for label in hypothesis.labels()
                   if label not in self.identification.references}
        hypothesis.rename_labels(mapping=mapping, copy=False)
        update_references(file, hypothesis, '@emb', self.identification.references)

        # FIXME what is this ?? (from anthony baseline)
        if not inputs.hasMoreData():
            pass

        print('\n\n')
        # always return True, it signals BEAT to continue processing
        return True


if __name__ == '__main__':

    # s = Segment(0, 1)
    # a = Annotation()
    # a[s] = 'foo'
    # t_ok = Time(0.5)
    # t_out = Time(100)
    # print(t_ok.in_segment(s))
    # print(t_out.in_segment(s))
    # print(t_ok.find_label(a))
    # print(t_out.find_label(a))
    # DEBUG : instantiate Algorithm
    algorithm = Algorithm()
    print('instantiated algorithm')
    references = get_references(algorithm.protocol,
                                algorithm.model['emb'],
                                subsets={'train', 'development'})
    algorithm.identification = algorithm.identification = KNearestSpeakers(references,
                                                   sad_scores='oracle',
                                                   scd_scores='@scd',
                                                   embedding='@emb',
                                                   metric=algorithm.metric,
                                                   evaluation_only=algorithm.evaluation_only,
                                                   confusion=False,
                                                   weigh=True)
    print('init identification pipeline ok')
    params = {
        'classifier': {'k': 28,
                       'threshold': 0.01946402570018635},
        'speech_turn_segmentation': algorithm.parameters['params']['speech_turn_segmentation']
    }
    algorithm.identification.instantiate(params)
    print('instantiate pipeline ok')
    protocol = get_protocol(algorithm.protocol)
    for file in protocol.test():
        uri = file['uri']
        print(uri)
        # compute SCD scores and embeddings
        scd, emb = Wrapper(algorithm.model['scd']), Wrapper(algorithm.model['emb'])
        file['scd'], file['emb'] = scd(file), emb(file)
        human_assisted_learning = True
        active = True
        if not human_assisted_learning:
            # TODO: unsupervised adaptation
            pass
        # 1. assign each segment to the closest reference if close enough
        # else tag with negative label
        hypothesis = algorithm.identification(file, use_threshold=True)

        # cluster < 0 (unk)
        unknown, hypothesis = hypothesis_to_unk(hypothesis)
        mapping = {label: str(label) for label in unknown.labels()}
        unknown.rename_labels(mapping=mapping, copy=False)
        unknown = algorithm.diarization.speech_turn_clustering(file,
                                                          unknown) if unknown else unknown
        # relabel new clusters so they don't mix with previous ones
        mapping = {label: f'{uri}#{label}' for label in unknown.labels()
                   if label not in algorithm.identification.references}
        unknown.rename_labels(mapping=mapping, copy=False)

        # update hypothesis with clustering results
        # unknown should then only be used to keep up with queried segments
        hypothesis = hypothesis.update(unknown)
        alliesAnnotation = AlliesAnnotation(hypothesis).to_hypothesis()
        # keep track of segments which come from same (resp. different) speakers
        positives, negatives = [], []
        # keep track of identification constraints
        must_link, cannot_link = hypothesis.empty(), hypothesis.empty()

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
            if active:
                # The system can send a question to the human in the loop
                # by using an object of type request
                # The request is the question asked to the system
                # find segment farthest from all existing clusters in unknown speakers
                query_speaker, farthest, centroid = get_farthest(file,
                                                                 unknown,
                                                                 '@emb',
                                                                 metric=algorithm.metric)
                time_1, time_2 = farthest.middle, centroid.middle
                # is this farthest segment in the right cluster ?
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
            break

        centroids = get_centroids(file, hypothesis, '@emb', metric='cosine')
        break


