import time
import numpy as np

cimport numpy as np

FLOATTYPE = np.float32

from graph_based_clustering cimport GraphBasedClustering
from graph_based_clustering import GraphBasedClustering

cdef class AverageLinkClustering(GraphBasedClustering):
  """
    Forms clusters by adding the edge which adds the minimum inter-cluster distence.
  """

  def __cinit__(self):
    self.calculated_distances = {}

  cdef void _cluster(self, G, num_clusters, distances):
    start_time = time.time()

    # Keep track of which cluster each vlmc is in
    clustering = {}
    for i, vlmc in enumerate(self.vlmcs):
      clustering[i] = [i]

    connections_to_make = len(self.vlmcs) - num_clusters

    for i in range(connections_to_make):
      edge_to_add = self._find_min_edge(clustering, distances)
      # If there are no more edges to add...
      if edge_to_add == -1:
        break
      (left, right, dist) = distances[edge_to_add]

      G.add_edge(self.vlmcs[int(left)], self.vlmcs[int(right)], weight=dist)

      clustering = self._merge_clusters(clustering, left, right)

    cluster_time = time.time() - start_time
    print("Cluster time: {} s".format(cluster_time))

  cdef int _find_min_edge(self, clustering, distances):
    min_edge_distance = np.inf
    min_edge = -1

    for i, edge in enumerate(distances):
      (left, right, _) = edge
      if clustering[int(left)] != clustering[int(right)]:
        added_dist = self._added_internal_distance_with_edge(edge, clustering, distances)
        if added_dist < min_edge_distance:
          min_edge_distance = added_dist
          min_edge = i

    return min_edge

  cdef double _added_internal_distance_with_edge(self, edge, clustering, distances):
    (left, right, dist) = edge
    left_cluster = clustering[int(left)]
    right_cluster = clustering[int(right)]

    merged_list = left_cluster + right_cluster
    merged_list.sort()
    key = tuple(merged_list)

    if key in self.calculated_distances:
      return self.calculated_distances[key]

    added_internal_distance = 0

    for from_cluster in left_cluster:
      for to_cluster in right_cluster:
        added_internal_distance += self.indexed_distances[from_cluster, to_cluster]

    #   Technically this should be len(left_cluster) * len(right_cluster), but,
    # unscientifically, this seems to work better
    final_distance = added_internal_distance / (len(left_cluster) + len(right_cluster))

    # Save the distance so we don't have to recalculate it.
    self.calculated_distances[key] = final_distance
    return final_distance

  cdef dict _merge_clusters(self, clustering, left, right):
    left_cluster = clustering[int(left)]
    right_cluster = clustering[int(right)]

    if len(left_cluster) < len(right_cluster):
      clustering = self._merge_clusters_(clustering, right_cluster, left_cluster)
    else:
      clustering = self._merge_clusters_(clustering, left_cluster, right_cluster)

    return clustering

  cdef dict _merge_clusters_(self, clustering, large_cluster, small_cluster):
    large_cluster.extend(small_cluster)
    for i in small_cluster:
      clustering[i] = large_cluster

    return clustering