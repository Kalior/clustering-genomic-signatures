from itertools import product
import networkx as nx
import numpy as np
cimport numpy as np
import random
import math

FLOATTYPE = np.float32
ctypedef np.float32_t FLOATTYPE_t

cdef class KMeans:
  cdef list vlmcs
  cdef public dict context_transition_to_array_index
  cdef dict vlmc_to_array_index
  cdef np.ndarray projected_vlmcs
  cdef int nbr_vlmcs
  cdef public int dimension

  def __cinit__(self, vlmcs):
    self.context_transition_to_array_index = {}
    self.vlmcs = vlmcs
    self.initialize_transition_to_index_dict()

    self.nbr_vlmcs = len(vlmcs)
    self.dimension = len(self.context_transition_to_array_index) * 4
    self.projected_vlmcs = np.zeros([self.nbr_vlmcs, self.dimension], dtype=FLOATTYPE)
    self.initialize_vlmc_to_index_dict()

  cdef initialize_vlmc_to_index_dict(self):
    self.vlmc_to_array_index = {}
    for i, vlmc in enumerate(self.vlmcs):
      self.projected_vlmcs[i, :] = self._vlmc_to_vector(vlmc, self.dimension)
      self.vlmc_to_array_index[vlmc] = i

  cdef initialize_transition_to_index_dict(self):
    contexts_to_use = set()
    for vlmc in self.vlmcs:
      contexts_to_use.update(vlmc.tree.keys())
    cdef int i = 0
    for context in contexts_to_use:
      self.context_transition_to_array_index[context] = {}
      for character in ["A", "C", "G", "T"]:
        self.context_transition_to_array_index[context][character] = i
        i += 1

  cdef np.ndarray _vlmc_to_vector(self, vlmc, dimension):
    cdef np.ndarray[FLOATTYPE_t, ndim=1] array = np.zeros(dimension, dtype=FLOATTYPE)
    for context in vlmc.tree:
      for character in ["A", "C", "G", "T"]:
        index = self.context_transition_to_array_index[context][character]
        array[index] = vlmc.tree[context][character]
    return array

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
    cdef np.ndarray[FLOATTYPE_t, ndim=2] distances = self._calculate_distances()
    return G, distances.mean()

  def create_graph(self, nbr_clusters, vlmc_index_to_cluster_index):
    G = nx.Graph()
    G.add_nodes_from(self.vlmcs)
    for i in range(nbr_clusters):
      # Add each cluster to the graph
      vlmc_indecis_in_current_cluster = [vlmc for vlmc, index in self.vlmc_to_array_index.items() if
                                         vlmc_index_to_cluster_index[index] == i]
      for x, y in product(vlmc_indecis_in_current_cluster, vlmc_indecis_in_current_cluster):
        G.add_edge(x, y)
    return G

  def update_centroid(self, centroids, centroid_index, vlmc_index_to_cluster_index):
    vlmc_indecis_in_current_cluster = [j for (j, cluster_index) in enumerate(vlmc_index_to_cluster_index) if
                                       cluster_index == centroid_index]
    if len(vlmc_indecis_in_current_cluster) > 0:
      new_centroid = np.zeros([1, self.dimension])
      for k in vlmc_indecis_in_current_cluster:
        new_centroid[0, :] = new_centroid[0, :] + self.projected_vlmcs[k, :]
      normalizing_factor = 1.0 / len(vlmc_indecis_in_current_cluster)
      new_centroid[0, :] = normalizing_factor * new_centroid[0, :]
      centroids[centroid_index, :] = new_centroid[0, :]

  def find_closest_centroid(self, centroids, vlmc_index):
    closest_centroid = -1
    min_distance = math.inf
    for centroid_index, _ in enumerate(centroids):
      distance_vector = self.projected_vlmcs[vlmc_index, :] - centroids[centroid_index, :]
      distance = np.linalg.norm(distance_vector)
      if (distance < min_distance):
        min_distance = distance
        closest_centroid = centroid_index
    return closest_centroid

  def initialize_centroids_randomly(self, nbr_clusters):
    centroid_indices = random.sample(range(self.nbr_vlmcs), nbr_clusters)
    centroids = np.zeros([nbr_clusters, self.dimension])
    for i, centroid_index in enumerate(centroid_indices):
      centroids[i, :] = self.projected_vlmcs[centroid_index, :]
    return centroids

  cdef np.ndarray[FLOATTYPE_t, ndim=2] _calculate_distances(self):
    cdef int num_vlmcs = len(self.vlmcs)
    cdef int num_distances = num_vlmcs * num_vlmcs
    cdef np.ndarray[FLOATTYPE_t, ndim=2] distances = np.zeros([num_distances, 3], dtype=FLOATTYPE)

    cdef FLOATTYPE_t dist, left_i_t, right_i_t
    for left_i, left in enumerate(self.vlmcs):
      for right_i, right in enumerate(self.vlmcs):
        if right != left:
          dist = self.distance(left, right)
          left_i_t = left_i
          right_i_t = right_i
          distances_index = left_i * num_vlmcs + right_i
          distances[distances_index, 0] = left_i_t
          distances[distances_index, 1] = right_i_t
          distances[distances_index, 2] = dist

    return distances

  cdef distance(self, left, right):
    left_vector_i = self.vlmc_to_array_index[left]
    right_vector_i = self.vlmc_to_array_index[right]
    left_vector = self.projected_vlmcs[left_vector_i, :]
    right_vector = self.projected_vlmcs[right_vector_i, :]
    return np.linalg.norm(left_vector - right_vector)
