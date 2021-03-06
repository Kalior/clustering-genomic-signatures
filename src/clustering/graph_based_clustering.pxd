cimport numpy as np

ctypedef np.float32_t FLOATTYPE_t

cdef class GraphBasedClustering:
  """
    Super class for every graph-based clustering method.
  """

  cdef list vlmcs
  cdef object d
  cdef str file_name
  cdef np.ndarray distances
  cdef np.ndarray indexed_distances
  cdef int created_clusters
  cdef object G
  cdef dict metadata
  cdef list merge_distances

  cpdef object cluster(self, clusters)

  cdef void _initialise_clusters(self)

  cdef void _cluster(self, num_clusters, distances)

  cdef tuple _find_min_edge(self)

  cdef void _merge_clusters(self, left, right)

  cdef void _make_fully_connected_components(self)

  cdef FLOATTYPE_t _find_distance_from_vlmc(self, v1, v2)

  cdef np.ndarray[FLOATTYPE_t, ndim = 2] _calculate_distances(self)
