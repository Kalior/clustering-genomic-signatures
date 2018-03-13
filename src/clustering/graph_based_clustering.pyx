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

          self.indexed_distances[left_i, right_i] = dist

    np.save(self.file_name, distances)

    return distances
