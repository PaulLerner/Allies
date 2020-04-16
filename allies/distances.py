"""Compute distances between embeddings"""

from pyannote.core.utils.distance import pdist
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(precision=2, suppress=True)

def get_distances(references, metric='cosine'):
    """
    Gets the distances between every speech turn embeddings for every speaker.
    """
    X, Y = [], []
    for speaker, embeddings in references.items():
        Y.extend([speaker for _ in embeddings])
        X.extend(embeddings)
    X,Y=np.array(X),np.array(Y)
    distances=squareform(pdist(X, metric=metric))
    same_speaker=squareform(pdist(np.array(Y), metric='equal'))
    return distances, same_speaker

def stats(distances):
    mean,std=np.mean(distances),np.std(distances)
    print(f'mean: {mean}\nstd: {std}')
    print(f'quartiles: {np.quantile(distances, [0,0.25,0.5,0.75,1.0])}')
    return mean

def get_thresholds(references, metric='cosine'):
    distances, same_speaker=get_distances(references, metric=metric)
    print("same speaker:")
    close = stats(distances[same_speaker])
    print("different speaker:")
    far = stats(distances[~same_speaker])
    n, bins, patches = plt.hist(distances[~same_speaker],density=True,label="different speaker",alpha=0.5)
    plt.hist(distances[same_speaker], bins=bins, density=True,label="same speaker",alpha=0.5)
    plt.title(f"Distributions of {metric} distances between the speech turns embeddings")
    plt.xlabel(f"{metric} distance")
    plt.ylabel("density")
    plt.show()
    return {
        "close":close,
        "far":far
    }
