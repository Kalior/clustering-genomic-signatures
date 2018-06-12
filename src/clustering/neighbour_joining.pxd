cimport numpy as np

ctypedef np.float32_t FLOATTYPE_t

from graph_based_clustering cimport GraphBasedClustering


cdef class NeighbourJoining(GraphBasedClustering):
  cdef dict neighbour_distances
  cdef set clusters

  cdef double _q_distance(self, left, right)
  cdef double _new_dist(self, left_dist, right_dist, last_dist)
