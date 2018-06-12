cimport numpy as np

ctypedef np.float32_t FLOATTYPE_t

from graph_based_clustering cimport GraphBasedClustering

cdef class MSTClustering(GraphBasedClustering):
  """
    Super class for mst graph-based clustering methods.
  """

  cdef dict clustering

  cdef void _cluster(self, num_clusters, distances)
  cdef void _create_mst(self, sorted_distances, connections_to_make)
  cdef tuple _find_smallest_unconnected_edge(self, sorted_distances, smallest_distance_index)
