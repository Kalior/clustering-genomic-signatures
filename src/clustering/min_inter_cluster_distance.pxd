cimport numpy as np

ctypedef np.float32_t FLOATTYPE_t

from graph_based_clustering cimport GraphBasedClustering


cdef class MinInterClusterDistance(GraphBasedClustering):
  cdef dict calculated_distances

  cdef int _find_min_edge(self, clustering, distances, distances_dict)
  cdef double _added_internal_distance_with_edge(self, distance, clustering, distances, distances_dict)
  cdef dict _merge_clusters(self, clustering, left, right)
  cdef dict _merge_clusters_(self, clustering, large_list, small_list)
  cdef double _find_distance(self, distances_dict, int left, int right)
