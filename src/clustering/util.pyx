import numpy as np
cimport numpy as np
FLOATTYPE = np.float32
ctypedef np.float32_t FLOATTYPE_t

cpdef public np.ndarray[FLOATTYPE_t, ndim=2] calculate_distances_within_vlmcs(vlmcs, d):
  cdef int num_vlmcs = len(vlmcs)
  cdef int num_distances = num_vlmcs * (num_vlmcs - 1)
  cdef np.ndarray[FLOATTYPE_t, ndim=2] distances = np.zeros([num_distances, 3], dtype=FLOATTYPE)
  cdef FLOATTYPE_t dist, left_i_t, right_i_t
  distances_index = 0
  for left_i, left in enumerate(vlmcs):
    for right_i, right in enumerate(vlmcs):
      if right != left:
        dist = d.distance(left, right)
        left_i_t = left_i
        right_i_t = right_i
        distances[distances_index, 0] = left_i_t
        distances[distances_index, 1] = right_i_t
        distances[distances_index, 2] = dist
        distances_index += 1
  return distances
