import networkx as nx

import numpy as np
cimport numpy as np
import time

FLOATTYPE = np.float32

cdef class GraphBasedClustering:
  """
    Super class for every graph-based clustering method.
  """
  def __cinit__(self, vlmcs, d):
    self.vlmcs = vlmcs
    self.d = d
    self.file_name = 'cluster_distances'

  cpdef tuple cluster(self, clusters):
    G = nx.Graph()
    G.add_nodes_from(self.vlmcs)

    start_time = time.time()

    cdef np.ndarray[FLOATTYPE_t, ndim=2] distances = self._calculate_distances()

    # cdef np.ndarray[FLOATTYPE_t, ndim=2] distances = np.load(self.file_name + '.npy')

    distance_time = time.time() - start_time
    print("Distance time: {} s".format(distance_time))

    self._cluster(G, clusters, distances)
    distance_mean = np.mean(distances, axis=None)

    return G, distance_mean

  cdef void _cluster(self, G, num_clusters, distances):
    return

  cdef np.ndarray[FLOATTYPE_t, ndim=2] _calculate_distances(self):
    cdef int num_vlmcs = len(self.vlmcs)
    cdef int num_distances = num_vlmcs * (num_vlmcs - 1)
    cdef np.ndarray[FLOATTYPE_t, ndim=2] distances = np.zeros([num_distances, 3], dtype=FLOATTYPE)

    cdef FLOATTYPE_t dist, left_i_t, right_i_t
    distances_index = 0
    for left_i, left in enumerate(self.vlmcs):
      for right_i, right in enumerate(self.vlmcs):
        if right != left:
          dist = self.d.distance(left, right)
          left_i_t = left_i
          right_i_t = right_i
          distances[distances_index, 0] = left_i_t
          distances[distances_index, 1] = right_i_t
          distances[distances_index, 2] = dist
          distances_index += 1

    np.save(self.file_name, distances)

    return distances
