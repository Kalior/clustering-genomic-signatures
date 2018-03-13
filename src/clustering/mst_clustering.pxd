cimport numpy as np

ctypedef np.float32_t FLOATTYPE_t

from graph_based_clustering cimport GraphBasedClustering

cdef class MSTClustering(GraphBasedClustering):
  """
    Super class for mst graph-based clustering methods.
  """

  cdef void _cluster(self, G, num_clusters, distances)
  cdef void _create_mst(self, sorted_distances, G)
  cdef tuple _find_smallest_unconnected_edge(self, sorted_distances, smallest_distance_index, clustering)
  cdef dict _merge_clusters(self, clustering, left, right)
  cdef tuple _find_most_inconsistent_edge(self, G, sorted_distances)
