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

  cdef double _frobenius_norm(self, original_vlmc, estimated_vlmc):
    # intersection:
    cdef list shared_contexts = [context for context in original_vlmc.tree if context in estimated_vlmc.tree]

    # union
    # cdef set shared_contexts = set(original_vlmc.tree.keys()).union(set(estimated_vlmc.tree.keys()))

      
    cdef np.ndarray[FLOATTYPE_t, ndim=2] original_matrix = self._create_matrix(original_vlmc, shared_contexts)
    cdef np.ndarray[FLOATTYPE_t, ndim=2] estimated_matrix = self._create_matrix(estimated_vlmc, shared_contexts)

    cdef np.ndarray[FLOATTYPE_t, ndim=2] frobenius_matrix = original_matrix - estimated_matrix

    return np.linalg.norm(frobenius_matrix, ord='fro') / len(shared_contexts)

  cdef np.ndarray[FLOATTYPE_t, ndim=2] _create_matrix(self, vlmc, contexts_to_use):
    cdef int rows = len(contexts_to_use)
    cdef int columns = len(self.alphabet)
    cdef np.ndarray[FLOATTYPE_t, ndim=2] matrix = np.empty((rows, columns), dtype=FLOATTYPE)

    cdef FLOATTYPE_t val
    cdef int i, j
    for i, context in enumerate(contexts_to_use):
      weight_factor = len(context)
      for j, character in enumerate(self.alphabet):
        if context in vlmc.tree:
          val = vlmc.tree[context][character] * weight_factor
        else:
          val = vlmc.tree[vlmc.get_context(context)][character] * weight_factor
        matrix[i, j] = val

    return matrix

  cdef list _get_leaf_contexts(self, vlmc):
    return list(filter(lambda c: self._is_leaf_context(c, vlmc), list(vlmc.tree.keys())))

  cdef bint _is_leaf_context(self, context, vlmc):
    possible_leaves = list(map(lambda c: context + c, self.alphabet))
    # leaf contexts are defined as having no children at all
    return all(map(lambda leaf: not leaf in vlmc.tree, possible_leaves))
