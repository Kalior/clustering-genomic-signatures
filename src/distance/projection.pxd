cimport numpy as np

cdef class Projection:
  cdef public dict context_transition_to_array_index
  cdef public int dimension
  cdef list vlmcs
  cdef initialize_transition_to_index_dict(self)
  cpdef distance(self, left, right)
  cdef np.ndarray vlmc_to_vector(self, vlmc)
