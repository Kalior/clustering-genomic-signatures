cdef class PSTMatching(object):

  cdef public double dissimilarity_weight

  def __cinit__(self, dissimilarity_weight):
    self.dissimilarity_weight = dissimilarity_weight

  cpdef double distance(self, left_vlmc, right_vlmc):
    cdef set union = set(left_vlmc.tree.keys()).union(set(right_vlmc.tree.keys()))
    cdef set intersection = set(left_vlmc.tree.keys()).intersection(set(right_vlmc.tree.keys()))
    distance = 0
    
    for state in union:
      probability_term = (1 - self.dissimilarity_weight) * self.probability_cost(state, left_vlmc, right_vlmc)
      dissimilarity_term = self.dissimilarity_weight * self.dissimilarity_cost(state, left_vlmc, right_vlmc)
      weight = self.state_weight(state, left_vlmc, right_vlmc)
      distance += weight * (probability_term + dissimilarity_term)
    return distance / len(intersection)


  cdef double state_weight(self, state, left_vlmc, right_vlmc):
    left_state = left_vlmc.get_context(state)
    right_state = right_vlmc.get_context(state)
    weight = (left_vlmc.occurrence_probability(left_state) + right_vlmc.occurrence_probability(right_state)) / 2
    return weight


  cdef double dissimilarity_cost(self, state, left_vlmc, right_vlmc):
    if state in left_vlmc.tree and state in right_vlmc.tree:
      return 0.0
    if state in left_vlmc.tree:
      return self._dissimilarity_cost(left_vlmc, right_vlmc, state)
    else:
      return self._dissimilarity_cost(right_vlmc, left_vlmc, state)


  cdef double _dissimilarity_cost(self, vlmc, vlmc_without_state, state):
    closest_state_in_other = vlmc_without_state.get_context(state)
    distance_difference = abs(len(closest_state_in_other) - len(state))
    max_len = max(len(closest_state_in_other), len(state))
    return distance_difference / max_len
        

  cdef double probability_cost(self, state, left_vlmc, right_vlmc):
    if state in left_vlmc.tree and state in right_vlmc.tree:
      probability_vector_difference = sum([abs(left_vlmc.tree[state][character] - right_vlmc.tree[state][character])
                                          for character in left_vlmc.alphabet])
      probability_cost = probability_vector_difference / 2
      return probability_cost
    else:
      return 0
