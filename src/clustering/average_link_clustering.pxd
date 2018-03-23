cimport numpy as np

ctypedef np.float32_t FLOATTYPE_t

from graph_based_clustering cimport GraphBasedClustering


cdef class AverageLinkClustering(GraphBasedClustering):
  cdef dict calculated_distances

  cdef int _find_min_edge(self, clustering, distances)
  cdef double _added_internal_distance_with_edge(self, distance, clustering, distances)
  cdef dict _merge_clusters(self, clustering, left, right)
  cdef dict _merge_clusters_(self, clustering, large_cluster, small_cluster)
