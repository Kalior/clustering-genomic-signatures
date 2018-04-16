cdef class StationaryDistribution(object):
  """
    Distance simply based on the stationary distribution of a, c, g, t of the VLMLCs
  """

  cpdef double distance(self, left_vlmc, right_vlmc):
    cdef dict left_stationary_prob = self._find_stationary_probability(left_vlmc)
    cdef dict right_stationary_prob = self._find_stationary_probability(right_vlmc)

    alphabet = left_vlmc.alphabet
    cdef double distance = sum([abs(left_stationary_prob[char_] - right_stationary_prob[char_])
                                  for char_ in alphabet])
    return distance

  cdef dict _find_stationary_probability(self, vlmc):
    sequence_length = 2000
    state_count = self._count_state_occourances(vlmc, sequence_length)

    char_probabilities = {}

    for char_ in vlmc.alphabet:
      prob = 0
      for key, value in state_count.items():
        prob_of_state = value / sequence_length
        prob_of_char = vlmc.tree[key][char_]
        prob += prob_of_char * prob_of_state

      char_probabilities[char_] = prob

    return char_probabilities

  cdef dict _count_state_occourances(self, vlmc, sequence_length):
    sequence = vlmc.generate_sequence(sequence_length, 500)
    state_count = {}

    for i in range(sequence_length):
      current_sequence = sequence[0:i][-vlmc.order:]
      matching_state = vlmc.get_context(current_sequence)
      if matching_state in state_count:
        state_count[matching_state] += 1
      else:
        state_count[matching_state] = 1

    return state_count
