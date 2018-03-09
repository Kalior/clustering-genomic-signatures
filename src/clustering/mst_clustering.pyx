import time
import numpy as np

cimport numpy as np

FLOATTYPE = np.float32
INTTYPE = np.int
ctypedef np.float32_t FLOATTYPE_t
ctypedef np.int_t INTTYPE_t


from graph_based_clustering cimport GraphBasedClustering
from graph_based_clustering import GraphBasedClustering


cdef class MSTClustering(GraphBasedClustering):
  """
    A clustering implementation which keeps the VLMCs as nodes in a graph.
    Adds the minimum distance for every node which isn't in the same cluster already.
  """
  cdef _cluster(self, G, num_clusters, distances):
    start_time = time.time()

    # Sort the array by the distances
    cdef np.ndarray[FLOATTYPE_t, ndim=2] sorted_distances = distances[distances[:,2].argsort()]

    sorting_time = time.time() - start_time
    start_time = time.time()

    # Keep track of which cluster each vlmc is in
    clustering = {}
    for i, vlmc in enumerate(self.vlmcs):
      clustering[i] = vlmc.name

    connections_to_make = len(self.vlmcs) - num_clusters
    # Keep track of the currently smallest index
    cdef int smallest_distance_index = 0
    for _ in range(connections_to_make):
      # Add an edge for the shortest distance
      # Take the smallest distance
      [left, right, dist] = sorted_distances[smallest_distance_index]
      smallest_distance_index += 1
      # Remove distances for pairs which are in the same cluster
      while clustering[left] == clustering[right]:
        [left, right, dist] = sorted_distances[smallest_distance_index]
        smallest_distance_index += 1

      G.add_edge(self.vlmcs[int(left)], self.vlmcs[int(right)], weight=dist)

      # Save this value as it may get overwritten during the for-loop below.
      rename_cluster = clustering[int(left)]

      # Point every node in the clusters to the same value.
      for key, value in clustering.items():
        if value == rename_cluster:
          clustering[key] = clustering[int(right)]

      if smallest_distance_index >= len(sorted_distances):
        break

    cluster_time = time.time() - start_time
    print("Sorting time: {} s\nCluster time: {} s".format(sorting_time, cluster_time))
