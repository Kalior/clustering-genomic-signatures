import networkx as nx

import numpy as np
cimport numpy as np
import time
from util import calculate_distances_within_vlmcs
from clustering_metrics import ClusteringMetrics

FLOATTYPE = np.float32

cdef class GraphBasedClustering:
  """
    Super class for every graph-based clustering method.
  """

  def __cinit__(self, vlmcs, d, metadata):
    self.vlmcs = vlmcs
    self.d = d
    self.file_name = 'cluster_distances'
    self.metadata = metadata
    self.indexed_distances = np.ndarray([len(vlmcs), len(vlmcs)], dtype=FLOATTYPE)
    self.merge_distances = []

    G = nx.Graph()
    G.add_nodes_from(self.vlmcs)

    start_time = time.time()
    self.distances = self._calculate_distances()
    distance_time = time.time() - start_time
    print("Distance time: {} s".format(distance_time))

    self.created_clusters = len(vlmcs)
    self.G = nx.Graph()
    self.G.add_nodes_from(self.vlmcs)

  cpdef object cluster(self, clusters):
    if clusters > self.created_clusters:
      self.G = nx.Graph()
      self.G.add_nodes_from(self.vlmcs)

    self._cluster(clusters, self.distances)

    self._make_fully_connected_components()

    self.created_clusters = clusters

    distance_mean = np.mean(self.distances, axis=None)
    metrics = ClusteringMetrics(self.G, distance_mean, self.indexed_distances,
                                self.vlmcs, self.metadata, self.merge_distances)
    return metrics

  cdef void _make_fully_connected_components(self):
    connected_components = nx.connected_components(self.G)
    for component in connected_components:
      for v1 in component:
        for v2 in component:
          self.G.add_edge(v1, v2, weight=self._find_distance_from_vlmc(v1, v2))

  cdef FLOATTYPE_t _find_distance_from_vlmc(self, v1, v2):
    v1_idx = self.vlmcs.index(v1)
    v2_idx = self.vlmcs.index(v2)
    return self.indexed_distances[v1_idx, v2_idx]

  cdef void _cluster(self, num_clusters, distances):
    for (left, right, dist) in distances:
      self.G.add_edge(self.vlmcs[int(left)], self.vlmcs[int(right)], weight=dist)

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

