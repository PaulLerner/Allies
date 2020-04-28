"""Compute distances between embeddings"""

from pyannote.core.utils.distance import pdist, cdist
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
import numpy as np
from pyannote.audio.features.wrapper import Wrapper, Wrappable

np.set_printoptions(precision=2, suppress=True)


def get_embeddings_per_speaker(current_file, hypothesis, model):
    """
    Gets the average speech turn embedding for every speaker and stores it in a dict.
    If a speech turn doesn't contains strictly an embedding then the 'center' mode is used for cropping,
    then the 'loose' mode. See `features.crop`

    Parameters
    ----------
    current_file: dict
        file as provided by pyannote protocol
    hypothesis : `Annotation`
    model: Wrappable
        Describes how raw speaker embeddings should be obtained.
        See pyannote.audio.features.wrapper.Wrapper documentation for details.

    Returns
    -------
    embeddings_per_speaker : `dict`
        {speaker:
            {segment: embedding}
        }
    """
    model = Wrapper(model)
    features = model(current_file)
    embeddings_per_speaker = {}
    for segment, track, label in hypothesis.itertracks(yield_label=True):
        # be more and more permissive until we have
        # at least one embedding for current speech turn
        for mode in ['strict', 'center', 'loose']:
            x = features.crop(segment, mode=mode)
            if len(x) > 0:
                break
        # skip speech turns so small we don't have any embedding for it
        if len(x) < 1:
            continue
        # average speech turn embedding
        x = np.mean(x, axis=0)
        embeddings_per_speaker.setdefault(label, {})
        embeddings_per_speaker[label][segment] = x
    return embeddings_per_speaker


def get_distances_per_speaker(current_file, hypothesis, model, metric='cosine'):
    """
    Gets the distances between every speech turn embeddings for every speaker.

    Parameters
    ----------
    current_file: dict
        file as provided by pyannote protocol
    hypothesis : `Annotation`
    model: Wrappable
        Describes how raw speaker embeddings should be obtained.
        See pyannote.audio.features.wrapper.Wrapper documentation for details.

    Returns
    -------
    distances_per_speaker : `dict` :
        {speaker:
            {segment: distance}
        }
        where distance corresponds to the distance between a speech turn and the centroid.
    """
    embeddings_per_speaker = get_embeddings_per_speaker(current_file, hypothesis, model)
    distances_per_speaker = {}
    for speaker, segments in embeddings_per_speaker.items():
        distances_per_speaker[speaker] = {}
        flat_embeddings = list(segments.values())
        distances = squareform(pdist(flat_embeddings, metric=metric))
        distances = np.mean(distances, axis=0)
        for i, segment in enumerate(segments.keys()):
            distances_per_speaker[speaker][segment] = distances[i]
    return distances_per_speaker

def get_centroids(current_file, annotation, model, metric='cosine'):
    """Finds the centroids of an annotation
    Parameters
    ----------
    current_file: dict
        file as provided by pyannote protocol
    annotation : `Annotation`
    model: Wrappable
        Describes how raw speaker embeddings should be obtained.
        See pyannote.audio.features.wrapper.Wrapper documentation for details.

    Returns
    -------
    centroids: Annotation
    """
    distances_per_speaker = get_distances_per_speaker(current_file,
                                                      annotation,
                                                      model,
                                                      metric=metric)

    # centroids
    centroids = annotation.empty()
    for speaker, segments in distances_per_speaker.items():
        centroid = min(segments)
        centroids[centroid, speaker] = speaker

    return centroids


def get_farthest(current_file, hypothesis, model, metric='cosine'):
    """Finds segment farthest from all existing clusters given :
    Parameters
    ----------
    current_file: dict
        file as provided by pyannote protocol
    hypothesis : `Annotation`
    model: Wrappable
        Describes how raw speaker embeddings should be obtained.
        See pyannote.audio.features.wrapper.Wrapper documentation for details.

    Returns
    -------
    speaker: str or int
        speaker/cluster which contains the farthest segment
    farthest: pyannote Segment
        farthest segment from all existing speakers/clusters
    centroid: pyannote Segment
        centroid of the cluster containing `farthest`
    """
    distances_per_speaker = get_distances_per_speaker(current_file,
                                                      hypothesis,
                                                      model,
                                                      metric=metric)

    # centroids
    min_per_speaker = {speaker: min(segments, key=segments.get)
                       for speaker, segments in distances_per_speaker.items()}
    # farthest segments
    max_per_speaker = {speaker: max(segments, key=segments.get)
                       for speaker, segments in distances_per_speaker.items()}
    # farthest of the farthests segments
    speaker = max(max_per_speaker, key=max_per_speaker.get)
    farthest, centroid = max_per_speaker[speaker], min_per_speaker[speaker]
    return speaker, farthest, centroid


def find_closest_to(to_segment, to_embedding, current_file, hypothesis, model,
                    metric='cosine'):
    """Finds segment closest to another one given :
    Parameters
    ----------
    to_segment: Segment,
        target segment
    to_embedding: np.ndarray,
        target embedding (which represents `to_segment`)
    current_file: dict
        file as provided by pyannote protocol
    hypothesis : `Annotation`
    model: Wrappable
        Describes how raw speaker embeddings should be obtained.
        See pyannote.audio.features.wrapper.Wrapper documentation for details.
    """
    embeddings_per_speaker = get_embeddings_per_speaker(current_file, hypothesis, model)
    segments, embeddings, speakers = [], [], []
    for speaker, speaker_segments in embeddings_per_speaker.items():
        for segment, embedding in speaker_segments.items():
            # of course the closest is self -> skip self
            if segment.intersects(to_segment):
                continue
            segments.append(segment)
            embeddings.append(embedding)
            speakers.append(speaker)
    # embedding must be 2D to use cdist
    to_embedding = to_embedding.reshape(1, -1)
    embeddings = np.array(embeddings)
    distance = cdist(embeddings, to_embedding, metric=metric)
    # reshape distance to the flat array it should be since to_embedding is 1D
    distance = distance.reshape(-1)
    i = np.argmin(distance)
    return speakers[i], segments[i]


def get_distances(references, metric='cosine'):
    """
    Gets the distances between every speech turn embeddings for every speaker
    from the references
    """
    X, Y = [], []
    for speaker, embeddings in references.items():
        Y.extend([speaker for _ in embeddings])
        X.extend(embeddings)
    X, Y = np.array(X), np.array(Y)
    distances = squareform(pdist(X, metric=metric))
    same_speaker = squareform(pdist(np.array(Y), metric='equal'))
    return distances, same_speaker


def stats(distances):
    mean, std = np.mean(distances), np.std(distances)
    print(f'mean: {mean:.2f}\nstd: {std:.2f}')
    print(f'quartiles: {np.quantile(distances, [0, 0.25, 0.5, 0.75, 1.0])}')
    return mean


def get_thresholds(references, metric='cosine'):
    distances, same_speaker = get_distances(references, metric=metric)
    print("same speaker:")
    close = stats(distances[same_speaker])
    print("different speaker:")
    far = stats(distances[~same_speaker])
    n, bins, patches = plt.hist(distances[~same_speaker], bins=50, density=True,
                                label="different speaker", alpha=0.5)
    plt.hist(distances[same_speaker], bins=bins, density=True, label="same speaker",
             alpha=0.5)
    plt.legend()
    title = f"Distributions of {metric} distances between the speech turns embeddings"
    plt.title(title)
    plt.xlabel(f"{metric} distance")
    plt.ylabel("density")
    plt.savefig(title.replace(' ', '_') + '.png')
    return {
        "close": close,
        "far": far
    }
