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
  def __cinit__(self, vlmcs, d):
    self.vlmcs = vlmcs
    self.d = d
    self.file_name = 'cluster_distances'
    self.indexed_distances = np.ndarray([len(vlmcs), len(vlmcs)], dtype=FLOATTYPE)

  cpdef object cluster(self, clusters):
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
    metrics = ClusteringMetrics(G, self.d, distance_mean, self.indexed_distances, self.vlmcs)
    return metrics

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

  cpdef dict silhouette_metric(self, G):
    connected_components = list(nx.connected_components(G))
    average_dist_to_own_component = {}
    min_dist_to_other_component = {}
    
    for component in connected_components:
      for v1 in component:
        minimum_average_distance_to_other_component = np.inf
        average_dist_to_own_component[v1.name] = self._same_component_average_distance_to_vlmcs(v1, component)

        # calculate distance to other components
        for other_component in connected_components:
          if component is other_component:
            continue
          average_distance = self._other_component_average_distance_to_vlmcs(v1, other_component)
          if average_distance < minimum_average_distance_to_other_component:
            minimum_average_distance_to_other_component = average_distance

        min_dist_to_other_component[v1.name] = minimum_average_distance_to_other_component
    silhouette = {}
    for v in G.nodes():
      s_i = ((min_dist_to_other_component[v.name] - average_dist_to_own_component[v.name]) /
             max(min_dist_to_other_component[v.name], average_dist_to_own_component[v.name]))
      silhouette[v.name] = s_i
    return silhouette
 
  cdef double _same_component_average_distance_to_vlmcs(self, v1, component):
    total_distance_to_vlmcs = 0
    for v2 in component:
      if v1 is v2:
        continue
      total_distance_to_vlmcs += self._find_distance_from_vlmc(v1, v2)

    average_internal_distance = 0
    nbr_other_elements_in_cluster = len(component) - 1
    if nbr_other_elements_in_cluster == 0:
       average_internal_distance = 0
    else:
       average_internal_distance = total_distance_to_vlmcs / nbr_other_elements_in_cluster
    return average_internal_distance

  cdef double _other_component_average_distance_to_vlmcs(self, v1, other_component):
    total_distance_to_vlmcs = 0
    average_external_distance = 0

    for v2 in other_component:
      total_distance_to_vlmcs += self._find_distance_from_vlmc(v1, v2)
    average_external_distance = total_distance_to_vlmcs / len(other_component)
    return average_external_distance
