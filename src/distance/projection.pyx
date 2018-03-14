import numpy as np
cimport numpy as np
FLOATTYPE = np.float32
ctypedef np.float32_t FLOATTYPE_t

cdef class Projection:
  cdef public dict context_transition_to_array_index
  cdef public int dimension
  def __cinit__(self, context_transition_to_array_index, dimension):
    self.context_transition_to_array_index = context_transition_to_array_index
    self.dimension = dimension

  cpdef distance(self, left, right):
    left_vector = self._vlmc_to_vector(left)
    right_vector = self._vlmc_to_vector(right)
    return np.linalg.norm(left_vector - right_vector)
  
  cdef np.ndarray _vlmc_to_vector(self, vlmc):
    cdef np.ndarray[FLOATTYPE_t, ndim=1] array = np.zeros(self.dimension, dtype=FLOATTYPE)
    for context in vlmc.tree:
      for character in ["A", "C", "G", "T"]:
        index = self.context_transition_to_array_index[context][character]
        array[index] = vlmc.tree[context][character]
    return array
