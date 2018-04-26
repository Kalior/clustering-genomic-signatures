import time
import numpy as np
from heapq import heappush, heapify, heappop
cimport numpy as np

FLOATTYPE = np.float32

from graph_based_clustering cimport GraphBasedClustering
from graph_based_clustering import GraphBasedClustering

cdef class AverageLinkClustering(GraphBasedClustering):
  """
    Forms clusters by adding the edge which adds the minimum inter-cluster distence.
  """

  def __cinit__(self):
    self.cluster_heaps = {}
    self.cluster_distances = {}
    self._initialise_clusters()

  cdef void _initialise_clusters(self):
    self.clustering = {}
    # Keep track of which cluster each vlmc is in
    for i, _ in enumerate(self.vlmcs):
      self.clustering[i] = [i]

    for left, _ in enumerate(self.vlmcs):
      heap = [(self.indexed_distances[left, right], (right,)) for right, _ in enumerate(self.vlmcs) if left != right]
      heapify(heap)
      self.cluster_heaps[(left,)] = heap

      distances = {(right,): self.indexed_distances[left, right] for right, _ in enumerate(self.vlmcs) if left != right}
      self.cluster_distances[(left,)] = distances


  cdef void _cluster(self, num_clusters, distances):
    start_time = time.time()

    if self.created_clusters < num_clusters:
      self._initialise_clusters()
      connections_to_make = len(self.vlmcs) - num_clusters
    else:
      connections_to_make = self.created_clusters - num_clusters

    for i in range(connections_to_make):
      left, right = self._find_min_edge()
      # If there are no more edges to add...
      if (-1,) in [left, right]:
        break

      self.G.add_edge(self.vlmcs[left[0]], self.vlmcs[right[0]],
          weight=self.indexed_distances[left[0], right[0]])
      self._merge_clusters(left, right)

    cluster_time = time.time() - start_time
    print("Cluster time: {} s".format(cluster_time))

  cdef tuple _find_min_edge(self):
    min_edge_distance = np.inf
    min_cluster_from = (-1,)
    min_cluster_to = (-1,)

    for key, heap in self.cluster_heaps.items():
      (distance, other_cluster_key) = heap[0]
      # Technically, invalid keys shouldn't be kept in the heap
      #  but removing them is O(n), and checking for existance in a
      #  hashmap is (almost always) O(1).
      while not other_cluster_key in self.cluster_distances[key]:
        heappop(heap)
        (distance, other_cluster_key) = heap[0]
      if distance < min_edge_distance:
        min_edge_distance = distance
        min_cluster_from = key
        min_cluster_to = other_cluster_key

    return min_cluster_from, min_cluster_to

  cdef void _merge_clusters(self, left, right):
    left_cluster = self.clustering[left[0]]
    right_cluster = self.clustering[right[0]]

    left_cluster_size = len(left_cluster)
    right_cluster_size = len(right_cluster)

    merged_list = left_cluster + right_cluster
    merged_list.sort()
    new_cluster_key = tuple(merged_list)

    self._merge_weights(left, right, left_cluster_size, right_cluster_size, new_cluster_key)
    self._merge_heaps(left, right, left_cluster_size, right_cluster_size, new_cluster_key)

    if len(left_cluster) < len(right_cluster):
      self._merge_clusters_(right_cluster, left_cluster)
    else:
      self._merge_clusters_(left_cluster, right_cluster)

  cdef void _merge_clusters_(self, large_cluster, small_cluster):
    large_cluster.extend(small_cluster)
    for i in small_cluster:
      self.clustering[i] = large_cluster

  cdef void _merge_weights(self, left, right, left_size, right_size, new_cluster_key):
    for key, heap in self.cluster_heaps.items():
      if not key in [left, right]:
        # Technically, the distances should be removed from the
        #  heaps as well, but this is O(n), which slows everything down
        #  to n^3 log n.  Instead, we keep track of which keys have been
        #  removed from the hash map.
        left_dist = self.cluster_distances[key].pop(left)
        right_dist = self.cluster_distances[key].pop(right)
        new_dist = self._new_dist(left_size, left_dist, right_size, right_dist)
        self.cluster_distances[key][new_cluster_key] = new_dist
        heappush(heap, (new_dist, new_cluster_key))

  cdef double _new_dist(self, left_size, left_dist, right_size, right_dist):
    new_dist = left_size * left_dist + right_size * right_dist
    new_dist /= left_size + right_size
    return new_dist

  cdef void _merge_heaps(self, left, right, left_size, right_size, new_cluster_key):
    # Only include distances which both clusters have keys for.
    keys = [k for k in self.cluster_distances[left].keys() if k in self.cluster_distances[right]]

    new_heap = []
    new_distances = {}

    for key in keys:
      left_dist = self.cluster_distances[left][key]
      right_dist = self.cluster_distances[right][key]
      new_dist = self._new_dist(left_size, left_dist, right_size, right_dist)
      new_distances[key] = new_dist
      heappush(new_heap, (new_dist, key))

    self.cluster_distances.pop(left)
    self.cluster_distances.pop(right)
    self.cluster_heaps.pop(left)
    self.cluster_heaps.pop(right)

    self.cluster_distances[new_cluster_key] = new_distances
    self.cluster_heaps[new_cluster_key] = new_heap
