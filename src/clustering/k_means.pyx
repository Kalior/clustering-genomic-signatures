from itertools import product
import networkx as nx
import numpy as np
cimport numpy as np
import random
import math


from util import calculate_distances_within_vlmcs, index_distances
from clustering_metrics import ClusteringMetrics
from distance.projection cimport Projection

INTTYPE = np.int32
ctypedef np.int32_t INTTYPE_t
FLOATTYPE = np.float32
ctypedef np.float32_t FLOATTYPE_t

cdef class KMeans:
  cdef list vlmcs
  cdef dict vlmc_to_array_index
  cdef np.ndarray projected_vlmcs
  cdef int nbr_vlmcs
  cdef Projection distance_function
  cdef dict metadata

  def __cinit__(self, vlmcs, d, metadata):
    self.vlmcs = vlmcs
    self.distance_function = d
    self.distance_function.set_vlmcs(vlmcs)
    self.nbr_vlmcs = len(vlmcs)
    self.projected_vlmcs = np.zeros(
        [self.nbr_vlmcs, self.distance_function.dimension], dtype=FLOATTYPE)
    self.initialize_vlmc_to_index_dict()
    self.metadata = metadata

  cdef initialize_vlmc_to_index_dict(self):
    self.vlmc_to_array_index = {}
    for i, vlmc in enumerate(self.vlmcs):
      self.projected_vlmcs[i, :] = self.distance_function.vlmc_to_vector(vlmc)
      self.vlmc_to_array_index[vlmc] = i

  cpdef object cluster(self, nbr_clusters):
    cdef np.ndarray[INTTYPE_t, ndim = 1] vlmc_index_to_cluster_index = np.zeros(self.nbr_vlmcs, dtype=INTTYPE)
    cdef np.ndarray[FLOATTYPE_t, ndim = 2] centroids = self.initialize_centroids_randomly(nbr_clusters)

    cdef bint some_vlmc_changed_cluster = True
    cdef int new_cluster = -1
    while some_vlmc_changed_cluster:
      # Assign vlmcs to closest centroid
      some_vlmc_changed_cluster = False
      for i, _ in enumerate(vlmc_index_to_cluster_index):
        new_cluster = self.find_closest_centroid(centroids, i)
        if new_cluster != vlmc_index_to_cluster_index[i]:
          vlmc_index_to_cluster_index[i] = new_cluster
          some_vlmc_changed_cluster = True

      # Update centroids
      for i in range(nbr_clusters):
        self.update_centroid(centroids, i, vlmc_index_to_cluster_index)

    G = self.create_graph(nbr_clusters, vlmc_index_to_cluster_index)
    cdef np.ndarray[FLOATTYPE_t, ndim = 2] distances = calculate_distances_within_vlmcs(self.vlmcs, self.distance_function)
    cdef np.ndarray[FLOATTYPE_t, ndim = 2] indexed_distances = index_distances(self.vlmcs, distances)

    metrics = ClusteringMetrics(G, distances.mean(),
                                indexed_distances, self.vlmcs, self.metadata)
    return metrics

  cdef object create_graph(self, nbr_clusters, vlmc_index_to_cluster_index):
    G = nx.Graph()
    G.add_nodes_from(self.vlmcs)
    for i in range(nbr_clusters):
      # Add each cluster to the graph
      vlmc_indices_in_current_cluster = [vlmc for vlmc, index in self.vlmc_to_array_index.items() if
                                         vlmc_index_to_cluster_index[index] == i]
      for x, y in product(vlmc_indices_in_current_cluster, vlmc_indices_in_current_cluster):
        G.add_edge(x, y)
    return G

  cdef void update_centroid(self, centroids, centroid_index, vlmc_index_to_cluster_index):
    vlmc_indices_in_current_cluster = [j for (j, cluster_index) in enumerate(vlmc_index_to_cluster_index) if
                                       cluster_index == centroid_index]
    if len(vlmc_indices_in_current_cluster) > 0:
      new_centroid = np.zeros([1, self.distance_function.dimension])
      for k in vlmc_indices_in_current_cluster:
        new_centroid[0, :] = new_centroid[0, :] + self.projected_vlmcs[k, :]
      normalizing_factor = 1.0 / len(vlmc_indices_in_current_cluster)
      new_centroid[0, :] = normalizing_factor * new_centroid[0, :]
      centroids[centroid_index, :] = new_centroid[0, :]

  cdef INTTYPE_t find_closest_centroid(self, centroids, vlmc_index):
    closest_centroid = -1
    min_distance = math.inf
    for centroid_index, _ in enumerate(centroids):
      distance_vector = self.projected_vlmcs[vlmc_index, :] - centroids[centroid_index, :]
      distance = np.linalg.norm(distance_vector)
      if (distance < min_distance):
        min_distance = distance
        closest_centroid = centroid_index
    return closest_centroid

  cdef np.ndarray[FLOATTYPE_t, ndim = 2] initialize_centroids_randomly(self, nbr_clusters):
    centroid_indices = random.sample(range(self.nbr_vlmcs), nbr_clusters)
    cdef np.ndarray[FLOATTYPE_t, ndim = 2] centroids = np.zeros([nbr_clusters, self.distance_function.dimension], dtype=FLOATTYPE)
    for i, centroid_index in enumerate(centroid_indices):
      centroids[i, :] = self.projected_vlmcs[centroid_index, :]
    return centroids

  cdef FLOATTYPE_t distance(self, left, right):
    left_vector_i = self.vlmc_to_array_index[left]
    right_vector_i = self.vlmc_to_array_index[right]
    left_vector = self.projected_vlmcs[left_vector_i, :]
    right_vector = self.projected_vlmcs[right_vector_i, :]
    return np.linalg.norm(left_vector - right_vector)
