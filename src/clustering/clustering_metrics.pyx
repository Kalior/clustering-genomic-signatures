import networkx as nx
import numpy as np
cimport numpy as np

FLOATTYPE = np.float32
ctypedef np.float32_t FLOATTYPE_t

cdef class ClusteringMetrics(object):
  """
  Class that calculates different metrics given a clustering of vlmcs.
  """
  cdef public object G
  cdef public double distance_mean
  cdef np.ndarray indexed_distances
  cdef public dict metadata
  cdef object d
  cdef list vlmcs

  def __cinit__(self, G, d, distance_mean, indexed_distances, vlmcs):
    self.G = G # the clustering of the vlmcs
    self.distance_mean = distance_mean
    self.indexed_distances = indexed_distances
    self.metadata = None # needs to be set after initialization
    self.d = d
    self.vlmcs = vlmcs

  cpdef double average_silhouette(self):
    silhouette = self.silhouette_metric()

    silhouette_values = [s for s in silhouette.values()]
    return sum(silhouette_values) / len(silhouette_values)

  cpdef dict silhouette_metric(self):
    connected_components = list(nx.connected_components(self.G))
    average_dist_to_own_component = {}
    min_dist_to_other_component = {}

    for component in connected_components:
      for v1 in component:
        minimum_average_distance_to_other_component = np.inf
        average_dist_to_own_component[v1.name] = self._same_component_average_distance_to_vlmcs(v1, component)

        # calculate distance to other components
        for other_component in connected_components:
          if component is other_component:
            continue
          average_distance = self._other_component_average_distance_to_vlmcs(v1, other_component)
          if average_distance < minimum_average_distance_to_other_component:
            minimum_average_distance_to_other_component = average_distance

        min_dist_to_other_component[v1.name] = minimum_average_distance_to_other_component
    silhouette = {}
    for v in self.G.nodes():
      s_i = ((min_dist_to_other_component[v.name] - average_dist_to_own_component[v.name]) /
             max(min_dist_to_other_component[v.name], average_dist_to_own_component[v.name]))
      silhouette[v.name] = s_i
    return silhouette

  cdef FLOATTYPE_t _same_component_average_distance_to_vlmcs(self, v1, component):
    total_distance_to_vlmcs = 0
    for v2 in component:
      if v1 is v2:
        continue
      total_distance_to_vlmcs += self._find_distance_from_vlmc(v1, v2)

    average_internal_distance = 0
    nbr_other_elements_in_cluster = len(component) - 1
    if nbr_other_elements_in_cluster == 0:
       average_internal_distance = 0
    else:
       average_internal_distance = total_distance_to_vlmcs / nbr_other_elements_in_cluster
    return average_internal_distance

  cdef FLOATTYPE_t _other_component_average_distance_to_vlmcs(self, v1, other_component):
    total_distance_to_vlmcs = 0
    average_external_distance = 0

    for v2 in other_component:
      total_distance_to_vlmcs += self._find_distance_from_vlmc(v1, v2)
    average_external_distance = total_distance_to_vlmcs / len(other_component)
    return average_external_distance

  cpdef FLOATTYPE_t average_distance_between_vlmcs(self, connected_component):
    distances = [self.d.distance(v1, v2) for v1 in connected_component for v2 in connected_component]
    average_distance = sum(distances) / len(distances)
    return average_distance

  cpdef double average_percent_same_taxonomy(self, taxonomy):
    average = 0
    connected_components = list(nx.connected_components(self.G))
    for connected_component in connected_components:
      average += self.percent_same_taxonomy(connected_component, taxonomy) * len(connected_component)

    return average / len(self.vlmcs)

  cpdef double percent_same_taxonomy(self, connected_component, taxonomy):
    size_of_component = len(connected_component)
    percent_of_same_family = sum(
      [self._number_in_taxonomy(vlmc, connected_component, taxonomy) for vlmc in connected_component]
    ) / (size_of_component ** 2)
    return percent_of_same_family

  cdef int _number_in_taxonomy(self, vlmc, vlmcs, taxonomy):
    number_of_same_taxonomy = len([other for other in vlmcs
                                 if self.metadata[other.name][taxonomy] == self.metadata[vlmc.name][taxonomy]])
    return number_of_same_taxonomy

  cdef FLOATTYPE_t _find_distance_from_vlmc(self, v1, v2):
    v1_idx = self.vlmcs.index(v1)
    v2_idx = self.vlmcs.index(v2)
    return self.indexed_distances[v1_idx, v2_idx]
