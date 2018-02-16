import numpy as np

import matplotlib.pyplot as plt


cdef class TransientDistribution(object):
  """
    Distance based on the different transient states of the vlmcs.
  """

  cdef list alphabet
  cdef int window_size
  cdef int window_step

  def __init__(self, window_size=500, window_step=50):
    # Assume this is the alphabet, only relevant case for us.
    self.alphabet = ['A', 'C', 'G', 'T']
    self.window_size = window_size
    self.window_step = window_step

  cpdef double distance(self, left_vlmc, right_vlmc):
    left_state_averages = self._find_transient_distriubutions(left_vlmc)
    right_state_averages = self._find_transient_distriubutions(right_vlmc)

    cdef double distance = sum([np.power(right_state_averages[k] - left_state_averages[k], 2)
                                   for k in left_vlmc.tree.keys()
                                   if k in left_state_averages and k in right_state_averages])
    # plt.show()
    return distance


  cdef _find_transient_distriubutions(self, vlmc):
    sequence_length = 10000

    state_list = self._get_state_sequence(vlmc, sequence_length)
    char_distributions, state_distributions = self._calculate_sliding_window_distibution(vlmc, state_list)
    # self._plot_sliding_window(vlmc, char_distributions, state_distributions)

    # char_averages = self._calculate_average(char_distributions, self.alphabet)
    state_averages = self._calculate_average(state_distributions, vlmc.tree.keys())

    # char_deviations = self._calculate_deviation(char_distributions, char_averages, self.alphabet)
    state_deviations = self._calculate_deviation(state_distributions, state_averages, vlmc.tree.keys())

    # self._print_average_and_deviation(char_averages, char_deviations, self.alphabet)
    # self._print_average_and_deviation(state_averages, state_deviations, vlmc.tree.keys())

    return state_averages


  cdef tuple _calculate_sliding_window_distibution(self, vlmc, state_list):
    windows = round((len(state_list) - self.window_size) / self.window_step)

    distributions = np.empty(windows, dtype=object)
    state_distributions = np.empty(windows, dtype=object)
    for i in range(windows):
      start = i * self.window_step
      end = start + self.window_size
      char_distribution, state_distribution = self._get_distribution(vlmc, state_list[start:end])
      distributions[i] = char_distribution
      state_distributions[i] = state_distribution

    return distributions, state_distributions

  cdef _plot_sliding_window(self, vlmc, char_distributions, state_distributions):
    fig, ax = plt.subplots(round(len(vlmc.tree.keys()) / 6) + 1, 6)
    for i, k in enumerate(vlmc.tree.keys()):
      col = i % ((len(vlmc.tree.keys()) / 6) + 1)
      row = i // ((len(vlmc.tree.keys()) / 6) + 1)
      ax[col, row].set_title(k)
      ax[col, row].plot([d[k] for d in state_distributions if k in d])

    fig, ax = plt.subplots(4)
    for i, k in enumerate(self.alphabet):
      ax[i].set_title(k)
      ax[i].plot([d[k] for d in char_distributions])

  cdef dict _calculate_average(self, distributions, keys):
    averages = {}

    for k in keys:
      averages[k] = sum([d[k] for d in distributions if k in d]) / len(distributions)

    return averages

  cdef dict _calculate_deviation(self, distributions, averages, keys):
    deviations = {}

    for k in keys:
      deviations[k] = np.sqrt(sum([np.power((d[k] - averages[k]), 2) for d in distributions if k in d]) / len(distributions))

    return deviations

  cdef _print_average_and_deviation(self, averages, deviations, keys):
    for k in keys:
      print("{:3}: average: {:.5f}  deviation: {:.5f}".format(k, averages[k], deviations[k]))

    print("")

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
