cimport numpy as np

ctypedef np.float32_t FLOATTYPE_t

from graph_based_clustering cimport GraphBasedClustering


cdef class AverageLinkClustering(GraphBasedClustering):
  cdef dict calculated_distances
  cdef dict clustering

  cdef void _initialise_clusters(self)
  cdef int _find_min_edge(self, distances)
  cdef double _added_internal_distance_with_edge(self, distance, distances)
  cdef dict _merge_clusters(self, left, right)
  cdef dict _merge_clusters_(self, large_cluster, small_cluster)
