import networkx as nx

import numpy as np
cimport numpy as np
import time
from util import calculate_distances_within_vlmcs

FLOATTYPE = np.float32

cdef class GraphBasedClustering:
  """
    Super class for every graph-based clustering method.
  """
  def __cinit__(self, vlmcs, d):
    self.vlmcs = vlmcs
    self.d = d
    self.file_name = 'cluster_distances'
    self.indexed_distances = np.ndarray([len(vlmcs), len(vlmcs)], dtype=FLOATTYPE)

  cpdef tuple cluster(self, clusters):
    G = nx.Graph()
    G.add_nodes_from(self.vlmcs)

    start_time = time.time()

    cdef np.ndarray[FLOATTYPE_t, ndim=2] distances = self._calculate_distances()

    # cdef np.ndarray[FLOATTYPE_t, ndim=2] distances = np.load(self.file_name + '.npy')

    distance_time = time.time() - start_time
    print("Distance time: {} s".format(distance_time))

    self._cluster(G, clusters, distances)

    self._make_fully_connected_components(G)

    distance_mean = np.mean(distances, axis=None)

    return G, distance_mean

  cdef void _make_fully_connected_components(self, G):
    connected_components = nx.connected_components(G)
    for component in connected_components:
      for v1 in component:
        for v2 in component:
          G.add_edge(v1, v2, weight=self._find_distance_from_vlmc(v1, v2))

  cdef FLOATTYPE_t _find_distance_from_vlmc(self, v1, v2):
    v1_idx = self.vlmcs.index(v1)
    v2_idx = self.vlmcs.index(v2)
    return self.indexed_distances[v1_idx, v2_idx]

  cdef void _cluster(self, G, num_clusters, distances):
    for (left, right, dist) in distances:
      G.add_edge(self.vlmcs[int(left)], self.vlmcs[int(right)], weight=dist)

  cdef np.ndarray[FLOATTYPE_t, ndim=2] _calculate_distances(self):
    cdef np.ndarray[FLOATTYPE_t, ndim=2] distances = calculate_distances_within_vlmcs(self.vlmcs, self.d)
    cdef int left_i = -1
    cdef int right_i = -1
    cdef FLOATTYPE_t dist = -1.0
  
    for column in distances:
      left_i = column[0]
      right_i = column[1]
      dist = column[2]
      self.indexed_distances[left_i, right_i] = dist

    np.save(self.file_name, distances)

    return distances
