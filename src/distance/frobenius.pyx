import numpy as np
cimport numpy as np
FLOATTYPE = np.float32
ctypedef np.float32_t FLOATTYPE_t

cdef class FrobeniusNorm(object):
  """
    Distance calculated by finding the transition matricies of the vlmcs, and
    calculating the frobenius norm of the difference.
  """

  cdef list alphabet

  def __init__(self, alphabet=['A', 'C', 'G', 'T']):
    self.alphabet = alphabet


  cpdef double distance(self, left_vlmc, right_vlmc):
    distance = self._frobenius_norm(left_vlmc, right_vlmc)
    return distance

  cdef double _frobenius_norm(self, left_vlmc, right_vlmc):
    # intersection:
    # cdef list shared_contexts = [context for context in left_vlmc.tree if context in right_vlmc.tree]

    # union
    cdef set shared_contexts = set(left_vlmc.tree.keys()).union(set(right_vlmc.tree.keys()))

    cdef np.ndarray[FLOATTYPE_t, ndim=2] left_matrix = self._create_matrix(left_vlmc, shared_contexts)
    cdef np.ndarray[FLOATTYPE_t, ndim=2] right_matrix = self._create_matrix(right_vlmc, shared_contexts)

    cdef np.ndarray[FLOATTYPE_t, ndim=2] frobenius_matrix = left_matrix - right_matrix

    return np.linalg.norm(frobenius_matrix, ord='fro') / np.sqrt(len(shared_contexts))

  cdef np.ndarray[FLOATTYPE_t, ndim=2] _create_matrix(self, vlmc, contexts_to_use):
    cdef int rows = len(contexts_to_use)
    cdef int columns = len(self.alphabet)
    cdef np.ndarray[FLOATTYPE_t, ndim=2] matrix = np.empty((rows, columns), dtype=FLOATTYPE)

    cdef FLOATTYPE_t val
    cdef int i, j
    for i, context in enumerate(contexts_to_use):
      for j, character in enumerate(self.alphabet):
        if context in vlmc.tree:
          val = vlmc.tree[context][character]
        else:
          val = vlmc.tree[vlmc.get_context(context)][character]
        matrix[i, j] = val

    return matrix
