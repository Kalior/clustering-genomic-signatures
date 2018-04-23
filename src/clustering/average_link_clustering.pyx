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
    self._initialise_clusters()

  cdef void _initialise_clusters(self):
    self.clustering = {}
    # Keep track of which cluster each vlmc is in
    for i, vlmc in enumerate(self.vlmcs):
      self.clustering[i] = [i]


  cdef void _cluster(self, num_clusters, distances):
    start_time = time.time()

    if self.created_clusters < num_clusters:
      self._initialise_clusters()
      connections_to_make = len(self.vlmcs) - num_clusters
    else:
      connections_to_make = self.created_clusters - num_clusters

    for i in range(connections_to_make):
      edge_to_add = self._find_min_edge(distances)
      # If there are no more edges to add...
      if edge_to_add == -1:
        break
      (left, right, dist) = distances[edge_to_add]

      self.G.add_edge(self.vlmcs[int(left)], self.vlmcs[int(right)], weight=dist)
      self._merge_clusters(left, right)

    cluster_time = time.time() - start_time
    print("Cluster time: {} s".format(cluster_time))

  cdef int _find_min_edge(self, distances):
    while True:
      i = np.random.random_integers(0, distances.shape[0])
      edge = distances[int(i)]
      (left, right, _) = edge
      if self.clustering[int(left)] != self.clustering[int(right)]:
        return i

  cdef double _added_internal_distance_with_edge(self, edge, distances):
    (left, right, dist) = edge
    left_cluster = self.clustering[int(left)]
    right_cluster = self.clustering[int(right)]

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
    final_distance = added_internal_distance / (len(left_cluster) * len(right_cluster))

    # Save the distance so we don't have to recalculate it.
    self.calculated_distances[key] = final_distance
    return final_distance

  cdef dict _merge_clusters(self, left, right):
    left_cluster = self.clustering[int(left)]
    right_cluster = self.clustering[int(right)]

    if len(left_cluster) < len(right_cluster):
      self._merge_clusters_(right_cluster, left_cluster)
    else:
      self._merge_clusters_(left_cluster, right_cluster)

  cdef dict _merge_clusters_(self, large_cluster, small_cluster):
    large_cluster.extend(small_cluster)
    for i in small_cluster:
      self.clustering[i] = large_cluster

