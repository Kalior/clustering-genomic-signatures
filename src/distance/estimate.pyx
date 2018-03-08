from scipy import stats
import numpy as np

from vlmc import VLMC
from . import NegativeLogLikelihood


cdef class EstimateVLMC(object):
  """
    Use the first vlmc to generate a sequence.  Estimate what the probabilities of the second
    model would be if it had generated that sequence.  Compare the estimated model with
    the second model, using a distance function.
  """
  cdef d
  alphabet = ['A', 'C', 'G', 'T']

  def __init__(self, d=NegativeLogLikelihood(1000)):
    self.d = d

  cpdef double distance(self, left_vlmc, right_vlmc):
    right_distance = self._assymmetric_distance(left_vlmc, right_vlmc)
    return right_distance
    # left_distance = self._assymmetric_distance(right_vlmc, left_vlmc)
    # return left_distance
    # return (right_distance + left_distance) / 2

  cdef double _assymmetric_distance(self, left_vlmc, right_vlmc):
    pre_sample_length = 500
    sequence_length = 100000

    right_sequence = right_vlmc.generate_sequence(sequence_length, pre_sample_length)

    left_transition_counters = self._count_events(left_vlmc, right_sequence)
    new_left_vlmc = self._create_vlmc_by_estimating_probabilities(
      self.alphabet, left_transition_counters)

    # distance = self.d.distance(left_vlmc, new_left_vlmc)
    distance = self._perform_stats_test(new_left_vlmc, left_vlmc, left_transition_counters)
    return distance

  cdef object _create_vlmc_by_estimating_probabilities(self, alphabet, transition_counters):
    cdef dict tree = {}
    for context, counts in transition_counters.items():
      count = sum([c for c in counts.values()])
      if count == 0:
        tree[context] = {'A': 0, 'C': 0, 'G': 0, 'T': 0}
      else:
        tree[context] = {}
        for character in alphabet:
          tree[context][character] = transition_counters[context][character] / count

    return VLMC(tree, "estimated")

  cdef dict _count_events(self, vlmc, sequence):
    cdef dict transition_counters = {}
    # Initialize counters
    for context in vlmc.tree.keys():
      transition_counters[context] = {}
      for character in self.alphabet:
        transition_counters[context][character] = 0

    sequence_so_far = ""
    for character in sequence:
      current_contexts = vlmc.get_all_contexts(sequence_so_far)
      for context in current_contexts:
        transition_counters[context][character] += 1

      sequence_so_far = sequence_so_far + character

    # return transition_counters
    transition_counters_with_pseudo = self._add_pseudo_counts(transition_counters)
    return transition_counters_with_pseudo

  cdef dict _add_pseudo_counts(self, transition_counters):
    for context, counts in transition_counters.items():
      if sum([c for c in counts.values()]) != 0 and any(c == 0 for c in counts.values()):
        transition_counters[context] = {t: c + 1 for t, c in transition_counters[context].items()}

    return transition_counters

  cdef double _perform_stats_test(self, estimated_vlmc, original_vlmc, transition_counters):
    expected_values = np.array([])
    observed_values = np.array([])
    for context in original_vlmc.tree.keys():
      times_visited_node = sum([c for c in transition_counters[context].values()])
      if times_visited_node > 0:
        transition_probabilitites = [(x, original_vlmc.tree[context][x]) for x in self.alphabet]

        # find probabilites that are greater than zero
        probs_without_zeros = [(char_, prob) for (char_, prob) in transition_probabilitites if prob > 0]
        # loop through all of these exept one (last)
        for character, probability_original_vlmc in probs_without_zeros[:-1]:
          probability_estimation = estimated_vlmc.tree[context][character]

          expected_frequency = probability_original_vlmc
          observed_frequency = probability_estimation
          expected_values = np.append(expected_values, expected_frequency)
          observed_values = np.append(observed_values, observed_frequency)

    statistic, p_value = stats.power_divergence(f_obs=observed_values, f_exp=expected_values, lambda_="pearson")
    # observed_mean = observed_values.mean()
    # observed_var = observed_values.var()

    # expected_mean = expected_values.mean()
    # expected_var = expected_values.var()

    # distance = np.linalg.norm(expected_values - observed_values)
    distance = statistic
    return distance
