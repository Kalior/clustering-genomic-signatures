import numpy as np

import matplotlib.pyplot as plt


cdef class TransientDistribution(object):
  """
    Distance based on the different transient states of the vlmcs.
  """

  cdef list alphabet

  def __init__(self):
    # Assume this is the alphabet, only relevant case for us.
    self.alphabet = ['A', 'C', 'G', 'T']

  cpdef double distance(self, left_vlmc, right_vlmc):
    self._find_transient_distriubutions(left_vlmc)
    self._find_transient_distriubutions(right_vlmc)

    # cdef double distance = sum([abs(left_stationary_prob[char_] - right_stationary_prob[char_])
                                  # for char_ in self.alphabet])
    plt.show()
    return 0.0


  cdef _find_transient_distriubutions(self, vlmc):
    sequence_length = 50000
    window_size = 5000
    window_step = 500

    windows = round((sequence_length - window_size) / window_step)

    state_list = self._get_state_sequence(vlmc, sequence_length)
    distributions = np.empty(windows, dtype=object)
    state_distributions = np.empty(windows, dtype=object)
    for i in range(windows):
      start = i * window_step
      end = start + window_size
      char_distribution, state_distribution = self._get_distribution(vlmc, state_list[start:end])
      distributions[i] = char_distribution
      state_distributions[i] = state_distribution

    fig, ax = plt.subplots(round(len(vlmc.tree.keys()) / 6) + 1, 6)
    for i, k in enumerate(vlmc.tree.keys()):
      col = i % ((len(vlmc.tree.keys()) / 6) + 1)
      row = i // ((len(vlmc.tree.keys()) / 6) + 1)
      ax[col, row].set_title(k)
      ax[col, row].plot([d[k] for d in state_distributions if k in d])

    fig, ax = plt.subplots(4)
    for i, k in enumerate(['A', 'C', 'G', 'T']):
      ax[i].set_title(k)
      ax[i].plot([d[k] for d in distributions])

  cdef tuple _get_distribution(self, vlmc, state_list):
    char_probabilities = {}
    state_count = {}

    for state in state_list:
      if state in state_count:
        state_count[state] += 1
      else:
        state_count[state] = 1

    state_probabilities = {}
    for key, value in state_count.items():
      state_probabilities[key] = value / len(state_list)

    for char_ in self.alphabet:
      prob = 0
      for key, value in state_count.items():
        prob_of_state = value / len(state_list)
        prob_of_char = vlmc.tree[key][char_]
        prob += prob_of_char * prob_of_state

      char_probabilities[char_] = prob

    return char_probabilities, state_probabilities

  cdef list _get_state_sequence(self, vlmc, sequence_length):
    sequence = vlmc.generate_sequence(sequence_length, 500)
    state_list = []

    for i in range(sequence_length):
      current_sequence = sequence[i-vlmc.order:i]
      matching_state = vlmc.get_context(current_sequence)
      state_list.append(matching_state)

    return state_list
