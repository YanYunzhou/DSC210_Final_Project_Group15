import numpy as np
from sklearn.neighbors import NearestNeighbors

def nearest_neighbor(src, dst):
    '''
    Find the nearest (Euclidean) neighbor in dst for each point in src
    '''

    neigh_k = NearestNeighbors(n_neighbors=1)
    neigh_k.fit(dst)
    distances, indices = neigh_k.kneighbors(src, return_distance=True)
    index_valid=np.where(distances.ravel()<0.5)
    return distances.ravel(), indices.ravel(),index_valid

