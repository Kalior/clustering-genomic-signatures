cimport numpy as np

ctypedef np.float32_t FLOATTYPE_t

cdef class GraphBasedClustering:
  """
    Super class for every graph-based clustering method.
  """

  cdef list vlmcs
  cdef object d
  cdef str file_name
  cdef np.ndarray indexed_distances

  cpdef tuple cluster(self, clusters)

  cdef void _cluster(self, G, num_clusters, distances)

  cdef void _make_fully_connected_components(self, G)

  cdef FLOATTYPE_t _find_distance_from_vlmc(self, v1, v2)

  cdef np.ndarray[FLOATTYPE_t, ndim=2] _calculate_distances(self)
