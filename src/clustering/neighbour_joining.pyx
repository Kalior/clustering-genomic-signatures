import time
import numpy as np
cimport numpy as np
from heapq import heapify

from skbio.tree import nj
from skbio import DistanceMatrix

FLOATTYPE = np.float32

from graph_based_clustering cimport GraphBasedClustering
from graph_based_clustering import GraphBasedClustering

cdef class NeighbourJoining(GraphBasedClustering):
  """
    Forms clusters by using the Neighbour joining clustering algorithm.
  """

  def __cinit__(self):
    self._initialise_clusters()

  # cdef void _cluster(self, num_clusters, distances):
  #   ids = [v.name for v in self.vlmcs]
  #   dm = DistanceMatrix(self.indexed_distances, ids)
  #   tree = nj(dm)
  #   print(tree.ascii_art())

  cdef void _initialise_clusters(self):
    # np.savetxt("distances", self.indexed_distances)
    self.clusters = {(i,) for i, _ in enumerate(self.vlmcs)}

    self.neighbour_distances = \
        {(i1,): {(i2,): self.indexed_distances[i1, i2]
                 for i2, _ in enumerate(self.vlmcs) if i1 != i2}
         for i1, _ in enumerate(self.vlmcs)}

  cdef tuple _find_min_edge(self):
    min_q = np.inf
    min_from = (-1,)
    min_to = (-1,)

    for left in self.clusters:
      for right in self.clusters:
        if left != right:
          q_distance = self._q_distance(left, right)

          if q_distance < min_q:
            min_from = left
            min_to = right
            min_q = q_distance

    return (min_from, min_to, min_q)

  cdef double _q_distance(self, left, right):
    number_of_clusters = len(self.clusters)
    q = (number_of_clusters - 2) * self.neighbour_distances[left][right]
    q -= sum([self.neighbour_distances[left][other]
              for other in self.clusters if left != other])
    q -= sum([self.neighbour_distances[right][other]
              for other in self.clusters if right != other])
    return q

  cdef double _new_dist(self, left_dist, right_dist, last_dist):
    return (left_dist + right_dist - last_dist) * 0.5

  cdef void _merge_clusters(self, left, right):
    merged_list = left + right
    new_cluster_key = tuple(merged_list)

    distance = self.neighbour_distances[left][right]

    self.clusters.remove(left)
    self.clusters.remove(right)

    for c in self.clusters:
      left_dist = self.neighbour_distances[c][left]
      right_dist = self.neighbour_distances[c][right]

      self.neighbour_distances[c].pop(left)
      self.neighbour_distances[c].pop(right)

      self.neighbour_distances[c][new_cluster_key] = \
          self._new_dist(left_dist, right_dist, distance)

    self.neighbour_distances[new_cluster_key] = {}
    for c in self.clusters:
      left_dist = self.neighbour_distances[left][c]
      right_dist = self.neighbour_distances[right][c]

      self.neighbour_distances[new_cluster_key][c] = \
          self._new_dist(left_dist, right_dist, distance)

    self.clusters.add(new_cluster_key)
    self.neighbour_distances.pop(left)
    self.neighbour_distances.pop(right)
