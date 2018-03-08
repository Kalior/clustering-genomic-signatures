import networkx as nx

import numpy as np
cimport numpy as np
import time

FLOATTYPE = np.float32
INTTYPE = np.int

cdef class GraphBasedClustering:
  """
    Super class for every graph-based clustering method.
  """
  def __cinit__(self, vlmcs, d):
    self.vlmcs = vlmcs
    self.d = d
    self.file_name = 'cluster_distances'

  cpdef tuple cluster(self, clusters):
    G = nx.Graph()
    G.add_nodes_from(self.vlmcs)

    start_time = time.time()

    cdef np.ndarray[FLOATTYPE_t, ndim=2] distances = self._calculate_distances()

    # cdef np.ndarray[FLOATTYPE_t, ndim=2] distances = np.load(self.file_name + '.npy')

    distance_time = time.time() - start_time
    print("Distance time: {} s".format(distance_time))

    self._cluster(G, clusters, distances)
    distance_mean = np.mean(distances, axis=None)

    return G, distance_mean

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


  cdef np.ndarray[FLOATTYPE_t, ndim=2] _calculate_distances(self):
    cdef int num_vlmcs = len(self.vlmcs)
    cdef int num_distances = num_vlmcs * num_vlmcs
    cdef np.ndarray[FLOATTYPE_t, ndim=2] distances = np.zeros([num_distances, 3], dtype=FLOATTYPE)

    cdef FLOATTYPE_t dist, left_i_t, right_i_t
    for left_i, left in enumerate(self.vlmcs):
      for right_i, right in enumerate(self.vlmcs):
        if right != left:
          dist = self.d.distance(left, right)
          left_i_t = left_i
          right_i_t = right_i
          distances_index = left_i * num_vlmcs + right_i
          distances[distances_index, 0] = left_i_t
          distances[distances_index, 1] = right_i_t
          distances[distances_index, 2] = dist

    np.save(self.file_name, distances)

    return distances
