import networkx as nx
import matplotlib.pyplot as plt
from distance import ACGTContent
from get_signature_metadata import get_metadata_for
import numpy as np
cimport numpy as np
import time

FLOATTYPE = np.float32
INTTYPE = np.int
ctypedef np.float32_t FLOATTYPE_t
ctypedef np.int_t INTTYPE_t

cdef class GraphBasedClustering(object):
  """
    A clustering implementation which keeps the VLMCs as nodes in a graph.
    There are several methods for creating the clusters:
    * Add edges where the distance between two nodes is less than some user-specified threshold, with
    one of the distance functions.
    * Add the minimum distance for every node which isn't in the same cluster already.
  """

  cdef double threshold
  cdef list vlmcs
  cdef d

  def __init__(self, threshold, vlmcs, d):
    self.threshold = threshold
    self.vlmcs = vlmcs
    self.d = d

  cpdef cluster(self, min_distance=True, draw_graph=False):
    G = nx.Graph()
    G.add_nodes_from(self.vlmcs)
    if min_distance:
      self._cluster_with_min_distance(G, 12)
    else:
      self._cluster_with_threshold(G)

    if draw_graph:
      self._draw_graph(G)
    self._print_connected_components(G)

  cdef _cluster_with_threshold(self, G):
    for vlmc in self.vlmcs:
      self._add_edges_from(G, vlmc)

  cdef _add_edges_from(self, G, vlmc):
    nearby_vlmcs = []
    for other in self.vlmcs:
      dist = self.d.distance(vlmc, other)
      if (dist < self.threshold):
        G.add_edge(vlmc, other, weight=dist)

  cdef _cluster_with_min_distance(self, G, num_clusters):
    start_time = time.time()
    cdef np.ndarray[FLOATTYPE_t, ndim=2] distances = self._calculate_distances()
    distance_time = time.time() - start_time
    start_time = time.time()

    cdef dict clustering = {}
    cdef FLOATTYPE_t i_t
    for i, vlmc in enumerate(self.vlmcs):
      i_t = i
      clustering[i_t] = vlmc.name

    cdef int connections_to_make = len(self.vlmcs) - num_clusters

    cdef int min_index
    for _ in range(connections_to_make):
      # Add an edge for the shortest distance
      # Finds the smallest distance (Would prefer if the distances were sorted instead, but numpy sorting was confusing)
      [_ , _, min_index] = np.argmin(distances, axis=0)
      [left, right, dist] = distances[min_index]
      distances = np.delete(distances, min_index, axis=0)
      # Remove distances for pairs which are in the same cluster
      while clustering[left] == clustering[right]:
        [_ , _, min_index] = np.argmin(distances, axis=0)
        [left, right, dist] = distances[min_index]
        distances = np.delete(distances, min_index, axis=0)

      G.add_edge(self.vlmcs[int(left)], self.vlmcs[int(right)], weight=dist)

      rename = clustering[left]

      # Point every node in the clusters to the same value.
      for key, value in clustering.items():
        if value == rename:
          clustering[key] = clustering[right]

      if distances.size == 0:
        break

    cluster_time = time.time() - start_time
    print("Distance time: {}s \n".format(distance_time))
    print("Cluster time: {}s \n".format(cluster_time))

  cdef np.ndarray[FLOATTYPE_t, ndim=2] _calculate_distances(self):
    cdef int num_vlmcs = len(self.vlmcs)
    cdef int num_distances = num_vlmcs * num_vlmcs
    cdef np.ndarray[FLOATTYPE_t, ndim=2] distances = np.zeros([num_distances, 3], dtype=FLOATTYPE)

    cdef FLOATTYPE_t dist, left_i_t, right_i_t
    for left_i, left in enumerate(self.vlmcs):
      for right_i, right in enumerate(self.vlmcs):
        if right != left:
          dist = self.d.distance(left, right)
          left_i_t = left_i
          right_i_t = right_i
          distances_index = left_i * num_vlmcs + right_i
          distances[distances_index, 0] = left_i_t
          distances[distances_index, 1] = right_i_t
          distances[distances_index, 2] = dist

    return distances

  def _draw_graph(self, G):
    plt.subplot(121)
    nx.draw_shell(G, with_labels=True, font_weight='bold')
    plt.show()

  def _print_connected_components(self, G):
    metadata = get_metadata_for([vlmc.name for vlmc in self.vlmcs])
    output = [self._component_string(connected, metadata)
              for connected in nx.connected_components(G)]

    print('\n\n'.join(output))

  def _component_string(self, connected, metadata):
    output = [self._output_line(metadata, vlmc) for vlmc in connected]
    return '\n'.join(output)

  def _output_line(self, metadata, vlmc):
    return "{:>55}  {:20} {:20}".format(
        metadata[vlmc.name]['species'],
        metadata[vlmc.name]['genus'],
        metadata[vlmc.name]['family'])
