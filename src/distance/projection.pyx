import numpy as np
cimport numpy as np
FLOATTYPE = np.float32
ctypedef np.float32_t FLOATTYPE_t

cdef class Projection:
  def __cinit__(self, vlmcs):
    self.vlmcs = vlmcs
    self.initialize_transition_to_index_dict()
    self.dimension = len(self.context_transition_to_array_index) * 4

  cdef initialize_transition_to_index_dict(self):
    self.context_transition_to_array_index = {}
    contexts_to_use = set()
    for vlmc in self.vlmcs:
      contexts_to_use.update(vlmc.tree.keys())
    cdef int i = 0
    for context in contexts_to_use:
      self.context_transition_to_array_index[context] = {}
      for character in ["A", "C", "G", "T"]:
        self.context_transition_to_array_index[context][character] = i
        i += 1
    
  cpdef distance(self, left, right):
    left_vector = self.vlmc_to_vector(left)
    right_vector = self.vlmc_to_vector(right)
    return np.linalg.norm(left_vector - right_vector)
  
  cdef np.ndarray vlmc_to_vector(self, vlmc):
    cdef np.ndarray[FLOATTYPE_t, ndim=1] array = np.zeros(self.dimension, dtype=FLOATTYPE)
    for context in vlmc.tree:
      for character in ["A", "C", "G", "T"]:
        index = self.context_transition_to_array_index[context][character]
        array[index] = vlmc.tree[context][character]
    return array
