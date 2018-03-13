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
  cdef void _cluster(self, G, num_clusters, distances):
    start_time = time.time()

    # Sort the array by the distances
    cdef np.ndarray[FLOATTYPE_t, ndim=2] sorted_distances = distances[distances[:,2].argsort()]

    sorting_time = time.time() - start_time
    start_time = time.time()

    self._create_mst(sorted_distances, G)

    for _ in range(num_clusters - 1):
      (left, right) = self._find_most_inconsistent_edge(G, sorted_distances)
      G.remove_edge(left, right)

    cluster_time = time.time() - start_time
    print("Sorting time: {} s\nCluster time: {} s".format(sorting_time, cluster_time))

  cdef void _create_mst(self, sorted_distances, G):
    # Keep track of which cluster each vlmc is in
    clustering = {}
    for i, vlmc in enumerate(self.vlmcs):
      clustering[i] = vlmc.name

    # Keep track of the currently smallest index
    cdef int smallest_distance_index = 0
    for _ in range(len(self.vlmcs) - 1):
      # Add an edge for the shortest distnce
      # Take the smallest distance
      smallest_distance_index, [left, right, dist] = \
        self._find_smallest_unconnected_edge(sorted_distances, smallest_distance_index, clustering)

      G.add_edge(self.vlmcs[int(left)], self.vlmcs[int(right)], weight=dist)

      clustering = self._merge_clusters(clustering, left, right)

      if smallest_distance_index >= len(sorted_distances):
        return

  cdef tuple _find_smallest_unconnected_edge(self, sorted_distances, smallest_distance_index, clustering):
    [left, right, dist] = sorted_distances[smallest_distance_index]
    smallest_distance_index += 1
    # Remove distances for pairs which are in the same cluster
    while clustering[left] == clustering[right]:
      [left, right, dist] = sorted_distances[smallest_distance_index]
      smallest_distance_index += 1

    return smallest_distance_index, [left, right, dist]

  cdef dict _merge_clusters(self, clustering, left, right):
    # Save this value as it may get overwritten during the for-loop below.
    rename_cluster = clustering[int(left)]

    # Point every node in the clusters to the same value.
    for key, value in clustering.items():
      if value == rename_cluster:
        clustering[key] = clustering[int(right)]

    return clustering

  cdef tuple _find_most_inconsistent_edge(self, G, sorted_distances):
    edges = G.edges(data=True)

    idx = np.argmax(np.array([e[2]['weight'] for e in edges]))
    (left, right, data) = edges[idx]
    return left, right
