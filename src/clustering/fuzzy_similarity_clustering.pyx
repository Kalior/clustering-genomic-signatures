import time
import numpy as np

cimport numpy as np

np.set_printoptions(threshold=np.nan)

FLOATTYPE = np.float32
ctypedef np.float32_t FLOATTYPE_t


from mst_clustering import MSTClustering
from mst_clustering cimport MSTClustering
from graph_based_clustering cimport GraphBasedClustering



cdef class FuzzySimilarityClustering(MSTClustering):
  """
    Clusters the the vlmcs based on the fuzzy similarity measure.
  """

  cdef np.ndarray[FLOATTYPE_t, ndim=2] _calculate_distances(self):
    cdef np.ndarray[FLOATTYPE_t, ndim=2] distances = GraphBasedClustering._calculate_distances(self)
    cdef np.ndarray[FLOATTYPE_t, ndim=2] sorted_distances = distances[distances[:,2].argsort()]

    k = 3
    rmax = 2
    alpha = 0.1
    cdef np.ndarray[FLOATTYPE_t, ndim=2] fuzzy_similarity_measures = \
      self._calculate_fuzzy_similarity_measures(sorted_distances, k, rmax, alpha)

    cdef np.ndarray[FLOATTYPE_t, ndim=2] fuzzy_similarity_measures_extended = \
      self._extend_format(fuzzy_similarity_measures)

    return fuzzy_similarity_measures_extended

  cdef np.ndarray[FLOATTYPE_t, ndim=2] _extend_format(self, fuzzy_similarity_measures):
    cdef int num_vlmcs = len(self.vlmcs)
    cdef np.ndarray[FLOATTYPE_t, ndim=2] fuzzy_measure = np.zeros([fuzzy_similarity_measures.size, 3], dtype=FLOATTYPE)
    cdef int i, j
    for i in range(len(self.vlmcs)):
      for j in range(len(self.vlmcs)):
        index = i * num_vlmcs + j
        fuzzy_measure[index, 0] = i
        fuzzy_measure[index, 1] = j
        fuzzy_measure[index, 2] = fuzzy_similarity_measures[i, j]

    return fuzzy_measure

  cdef np.ndarray[FLOATTYPE_t, ndim=2] _calculate_fuzzy_similarity_measures(self, distances, k, rmax, alpha):
    cdef np.ndarray[FLOATTYPE_t, ndim=2]  fuzzy_similarity_measures = np.zeros([len(self.vlmcs), len(self.vlmcs)], dtype=FLOATTYPE)
    for r in range(1, rmax + 1):
      for i, left in enumerate(self.vlmcs):
        for j, right in enumerate(self.vlmcs):
          fuzzy_similarity_measures[i, j] = (1 - alpha) * fuzzy_similarity_measures[i, j] + \
            alpha * self._fuzzy_similarity(i, j, k, r, distances)

    return fuzzy_similarity_measures

  cdef _fuzzy_similarity(self, i, j, k, r, distances):
    i_neighbours = self._k_nearest_neighbours(i, k * r, distances)
    j_neighbours = self._k_nearest_neighbours(j, k * r, distances)
    # print(i_neighbours)
    # print(j_neighbours)
    shared_neighbours_idx = np.where(np.equal(i_neighbours, j_neighbours))
    shared_neighbours = i_neighbours[shared_neighbours_idx]

    denominator = 2 * k * r - shared_neighbours.size

    return - shared_neighbours.size / denominator

  cdef _k_nearest_neighbours(self, i, k, distances):
    neighbour_idx = np.where( distances[:, 0] == i )
    neighbours = distances[neighbour_idx]
    return neighbours[0:k]
