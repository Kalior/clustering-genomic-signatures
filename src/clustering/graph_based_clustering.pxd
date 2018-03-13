cimport numpy as np

ctypedef np.float32_t FLOATTYPE_t

cdef class GraphBasedClustering:
  """
    Super class for every graph-based clustering method.
  """

  cdef list vlmcs
  cdef d
  cdef str file_name

  cpdef tuple cluster(self, clusters)

  cdef void _cluster(self, G, num_clusters, distances)

  cdef np.ndarray[FLOATTYPE_t, ndim=2] _calculate_distances(self)
