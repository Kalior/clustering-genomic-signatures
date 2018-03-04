import numpy as np
cimport numpy as np
FLOATTYPE = np.float32
ctypedef np.float32_t FLOATTYPE_t

cdef class FrobeniusNorm(object):
  """
    Distance simply based on the amount of a, c, g, t content in the strings.
    (Should be stored as transitions from the root node).
  """

  cdef list alphabet
  cdef public np.ndarray weight_parameters
  
  def __init__(self, weight_parameters ,alphabet=['A', 'C', 'G', 'T']):
    self.alphabet = alphabet
    self.weight_parameters = weight_parameters


  cpdef double distance(self, left_vlmc, right_vlmc):
    distance = self._frobenius_norm(left_vlmc, right_vlmc)
    return distance

  cdef double _frobenius_norm(self, original_vlmc, estimated_vlmc):
    cdef double norm = 0.0

    cdef np.ndarray[FLOATTYPE_t, ndim=2] original_matrix = self._create_matrix(original_vlmc)
    cdef np.ndarray[FLOATTYPE_t, ndim=2] estimated_matrix = self._create_matrix(estimated_vlmc)

    cdef np.ndarray[FLOATTYPE_t, ndim=2] frobenius_matrix = original_matrix - estimated_matrix

    return np.linalg.norm(frobenius_matrix, ord='fro')

  cdef np.ndarray[FLOATTYPE_t, ndim=2] _create_matrix(self, vlmc):
    leaf_contexts = vlmc.tree.keys()
    cdef int rows = len(leaf_contexts)
    cdef int columns = len(self.alphabet)
    cdef np.ndarray[FLOATTYPE_t, ndim=2] matrix = np.empty((rows, columns), dtype=FLOATTYPE)

    cdef FLOATTYPE_t val
    cdef int index = 0
    for i, context in enumerate(leaf_contexts):
      for j, character in enumerate(self.alphabet):
        val = vlmc.tree[context][character] * self.weight_parameters[index]
        matrix[i, j] = val
      index += 1

    return matrix

  cdef list _get_leaf_contexts(self, vlmc):
    return list(filter(lambda c: self._is_leaf_context(c, vlmc), list(vlmc.tree.keys())))

  cdef bint _is_leaf_context(self, context, vlmc):
    possible_leaves = list(map(lambda c: context + c, self.alphabet))
    # leaf contexts are defined as having no children at all
    return all(map(lambda leaf: not leaf in vlmc.tree, possible_leaves))

