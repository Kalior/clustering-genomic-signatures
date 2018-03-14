from itertools import product
import networkx as nx
import numpy as np
cimport numpy as np
import random
import math
from util import calculate_distances_within_vlmcs
from distance.projection cimport Projection

FLOATTYPE = np.float32
ctypedef np.float32_t FLOATTYPE_t

cdef class KMeans:
  cdef list vlmcs
  cdef dict vlmc_to_array_index
  cdef np.ndarray projected_vlmcs
  cdef int nbr_vlmcs
  cdef Projection distance_function

  def __cinit__(self, vlmcs, d):
    self.vlmcs = vlmcs
    self.distance_function = d
    self.nbr_vlmcs = len(vlmcs)
    self.projected_vlmcs = np.zeros([self.nbr_vlmcs, self.distance_function.dimension], dtype=FLOATTYPE)
    self.initialize_vlmc_to_index_dict()

  cdef initialize_vlmc_to_index_dict(self):
    self.vlmc_to_array_index = {}
    for i, vlmc in enumerate(self.vlmcs):
      self.projected_vlmcs[i, :] = self.distance_function.vlmc_to_vector(vlmc)
      self.vlmc_to_array_index[vlmc] = i

  cpdef tuple cluster(self, nbr_clusters):
    vlmc_index_to_cluster_index = np.zeros(self.nbr_vlmcs)
    centroids = self.initialize_centroids_randomly(nbr_clusters)

    some_vlmc_changed_cluster = True
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
    cdef np.ndarray[FLOATTYPE_t, ndim=2] distances = calculate_distances_within_vlmcs(self.vlmcs, self.distance_function)
    return G, distances.mean()

  cdef create_graph(self, nbr_clusters, vlmc_index_to_cluster_index):
    G = nx.Graph()
    G.add_nodes_from(self.vlmcs)
    for i in range(nbr_clusters):
      # Add each cluster to the graph
      vlmc_indecis_in_current_cluster = [vlmc for vlmc, index in self.vlmc_to_array_index.items() if
                                         vlmc_index_to_cluster_index[index] == i]
      for x, y in product(vlmc_indecis_in_current_cluster, vlmc_indecis_in_current_cluster):
        G.add_edge(x, y)
    return G

  cdef update_centroid(self, centroids, centroid_index, vlmc_index_to_cluster_index):
    vlmc_indecis_in_current_cluster = [j for (j, cluster_index) in enumerate(vlmc_index_to_cluster_index) if
                                       cluster_index == centroid_index]
    if len(vlmc_indecis_in_current_cluster) > 0:
      new_centroid = np.zeros([1, self.distance_function.dimension])
      for k in vlmc_indecis_in_current_cluster:
        new_centroid[0, :] = new_centroid[0, :] + self.projected_vlmcs[k, :]
      normalizing_factor = 1.0 / len(vlmc_indecis_in_current_cluster)
      new_centroid[0, :] = normalizing_factor * new_centroid[0, :]
      centroids[centroid_index, :] = new_centroid[0, :]

  cdef find_closest_centroid(self, centroids, vlmc_index):
    closest_centroid = -1
    min_distance = math.inf
    for centroid_index, _ in enumerate(centroids):
      distance_vector = self.projected_vlmcs[vlmc_index, :] - centroids[centroid_index, :]
      distance = np.linalg.norm(distance_vector)
      if (distance < min_distance):
        min_distance = distance
        closest_centroid = centroid_index
    return closest_centroid

  cdef initialize_centroids_randomly(self, nbr_clusters):
    centroid_indices = random.sample(range(self.nbr_vlmcs), nbr_clusters)
    centroids = np.zeros([nbr_clusters, self.distance_function.dimension])
    for i, centroid_index in enumerate(centroid_indices):
      centroids[i, :] = self.projected_vlmcs[centroid_index, :]
    return centroids

  cdef distance(self, left, right):
    left_vector_i = self.vlmc_to_array_index[left]
    right_vector_i = self.vlmc_to_array_index[right]
    left_vector = self.projected_vlmcs[left_vector_i, :]
    right_vector = self.projected_vlmcs[right_vector_i, :]
    return np.linalg.norm(left_vector - right_vector)
