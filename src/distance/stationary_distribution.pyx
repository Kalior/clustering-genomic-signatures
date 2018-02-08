cdef class StationaryDistribution(object):
  """
    Distance simply based on the stationary distribution of a, c, g, t of the VLMLCs
  """

  cdef list alphabet

  def __init__(self):
    # Assume this is the alphabet, only relevant case for us.
    self.alphabet = ['A', 'C', 'G', 'T']

  cpdef double distance(self, left_vlmc, right_vlmc):
    cdef dict left_stationary_prob = self._find_stationary_probability(left_vlmc)
    cdef dict right_stationary_prob = self._find_stationary_probability(right_vlmc)

    cdef double distance = sum([abs(left_stationary_prob[char_] - right_stationary_prob[char_])
                                  for char_ in self.alphabet])
    return distance

  cdef dict _find_stationary_probability(self, vlmc):
    sequence_length = 2000
    state_count = self._count_state_occourances(vlmc, sequence_length)

    char_probabilities = {}

    for char_ in self.alphabet:
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

    for i in range(len(sequence)):
      current_sequence = sequence[0:i][-vlmc.order:]
      matching_state = self._find_matching_state(vlmc.tree, current_sequence)
      if matching_state in state_count:
        state_count[matching_state] += 1
      else:
        state_count[matching_state] = 1

    return state_count

  cdef str _find_matching_state(self, tree, sequence):
    cdef str reverse_seq = sequence[::-1]
    cdef int depth = 0
    cdef str current_node = ""
    # Search down the tree for either the order of the vlmc, or the length of the string
    while current_node in tree and depth < len(reverse_seq):
      depth += 1
      current_node = reverse_seq[0:depth]
    return current_node


