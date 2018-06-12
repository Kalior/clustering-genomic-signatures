import time
import numpy as np

cimport numpy as np

FLOATTYPE = np.float32

from graph_based_clustering import GraphBasedClustering


cdef class MSTClustering(GraphBasedClustering):
  """
    A clustering implementation which keeps the VLMCs as nodes in a graph.
    Adds the minimum distance for every node which isn't in the same cluster already.
  """

  def __cinit__(self):
    self._initialise_clusters()

  cdef void _initialise_clusters(self):
    self.merge_distances = []
    self.clustering = {}
    # Keep track of which cluster each vlmc is in
    for i, vlmc in enumerate(self.vlmcs):
      self.clustering[i] = vlmc.name

  cdef void _cluster(self, num_clusters, distances):
    start_time = time.time()

    # Sort the array by the distances
    cdef np.ndarray[FLOATTYPE_t, ndim = 2] sorted_distances = distances[distances[:, 2].argsort()]

    sorting_time = time.time() - start_time
    start_time = time.time()

    if self.created_clusters < num_clusters:
      connections_to_make = len(self.vlmcs) - num_clusters
      self._initialise_clusters()
    else:
      connections_to_make = self.created_clusters - num_clusters

    self._create_mst(sorted_distances, connections_to_make)

    cluster_time = time.time() - start_time
    print("Sorting time: {} s\nCluster time: {} s".format(sorting_time, cluster_time))

  cdef void _create_mst(self, sorted_distances, connections_to_make):
    # Keep track of the currently smallest index
    cdef int smallest_distance_index = 0
    for i in range(connections_to_make):
      # Add an edge for the shortest distnce
      # Take the smallest distance
      smallest_distance_index, [left, right, dist] = \
          self._find_smallest_unconnected_edge(sorted_distances, smallest_distance_index)

      self.merge_distances.append(dist)

      self.G.add_edge(self.vlmcs[int(left)], self.vlmcs[int(right)], weight=dist)

      self._merge_clusters(left, right)

      if smallest_distance_index >= len(sorted_distances):
        return

  cdef tuple _find_smallest_unconnected_edge(self, sorted_distances, smallest_distance_index):
    [left, right, dist] = sorted_distances[smallest_distance_index]
    smallest_distance_index += 1
    # Remove distances for pairs which are in the same cluster
    while self.clustering[left] == self.clustering[right]:
      [left, right, dist] = sorted_distances[smallest_distance_index]
      smallest_distance_index += 1

    return smallest_distance_index, [left, right, dist]

  cdef void _merge_clusters(self, left, right):
    # Save this value as it may get overwritten during the for-loop below.
    rename_cluster = self.clustering[int(left)]

    # Point every node in the clusters to the same value.
    for key, value in self.clustering.items():
      if value == rename_cluster:
        self.clustering[key] = self.clustering[int(right)]
