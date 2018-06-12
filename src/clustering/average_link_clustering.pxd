cimport numpy as np

ctypedef np.float32_t FLOATTYPE_t

from graph_based_clustering cimport GraphBasedClustering


cdef class AverageLinkClustering(GraphBasedClustering):
  cdef dict cluster_distances
  cdef dict cluster_heaps
  cdef dict clustering

  cdef void _merge_clusters_(self, large_cluster, small_cluster)
  cdef void _merge_weights(self, left, right, left_size, right_size, new_cluster_key)
  cdef double _new_dist(self, left_size, left_dist, right_size, right_dist)
  cdef void _merge_heaps(self, left, right, left_size, right_size, new_cluster_key)
