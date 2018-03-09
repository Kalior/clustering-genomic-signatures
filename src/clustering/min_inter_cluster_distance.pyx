import time
import numpy as np

cimport numpy as np

FLOATTYPE = np.float32
INTTYPE = np.int
ctypedef np.float32_t FLOATTYPE_t
ctypedef np.int_t INTTYPE_t


from graph_based_clustering cimport GraphBasedClustering
from graph_based_clustering import GraphBasedClustering

import sys

cdef class MinInterClusterDistance(GraphBasedClustering):
  """
    Forms clusters by adding the edge which adds the minimum inter-cluster distence.
  """
  cdef _cluster(self, G, num_clusters, distances):
    start_time = time.time()

    distances_dict = {(d[0], d[1]): d[2] for d in distances}

    # Keep track of which cluster each vlmc is in
    clustering = {}
    for i, vlmc in enumerate(self.vlmcs):
      clustering[i] = [i]

    connections_to_make = len(self.vlmcs) - num_clusters
    # Keep track of the currently smallest index
    for i in range(connections_to_make):
      edge_to_add = self._find_min_edge(clustering, distances, distances_dict)
      # If there are no more edges to add...
      if edge_to_add == -1:
        break
      (left, right, dist) = distances[edge_to_add]

      G.add_edge(self.vlmcs[int(left)], self.vlmcs[int(right)], weight=dist)

      clustering = self._merge_clusters(clustering, left, right)

    cluster_time = time.time() - start_time
    print("Cluster time: {} s".format(cluster_time))

  cdef int _find_min_edge(self, clustering, distances, distances_dict):
    min_edge_distance = np.inf
    min_edge = -1

    for i, distance in enumerate(distances):
      (left, right, _) = distance
      if clustering[int(left)] != clustering[int(right)]:
        added_dist = self._added_internal_distance_with_edge(distance, clustering, distances, distances_dict)
        if added_dist < min_edge_distance:
          min_edge_distance = added_dist
          min_edge = i

    return min_edge

  cdef double _added_internal_distance_with_edge(self, distance, clustering, distances, distances_dict):
    (left, right, dist) = distance
    added_internal_distance = dist
    left_list = clustering[int(left)]
    right_list = clustering[int(right)]
    for from_cluster in left_list:
      for to_cluster in right_list:
        added_internal_distance += self._find_distance(distances_dict, from_cluster, to_cluster)

    #   Technically this should be len(left_list) * len(right_list), but,
    # unscientifically, this seems to work better
    return added_internal_distance / (len(left_list) + len(right_list))

  cdef dict _merge_clusters(self, clustering, left, right):
    left_list = clustering[int(left)]
    right_list = clustering[int(right)]

    if len(left_list) < len(right_list):
      clustering = self._merge_clusters_(clustering, right_list, left_list)
    else:
      clustering = self._merge_clusters_(clustering, left_list, right_list)

    return clustering

  cdef dict _merge_clusters_(self, clustering, large_list, small_list):
    large_list.extend(small_list)
    for i in small_list:
      clustering[i] = large_list

    return clustering

  cdef double _find_distance(self, distances_dict, int left, int right):
    if (left, right) in distances_dict:
      return distances_dict[(left, right)]
    raise KeyError("VLMC pair has no distance calculated between them.")

