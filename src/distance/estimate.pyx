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
    left_distance = self._assymmetric_distance(right_vlmc, left_vlmc)
    return (right_distance + left_distance) / 2

  cdef double _assymmetric_distance(self, left_vlmc, right_vlmc):
    pre_sample_length = 0
    sequence_length = 50000

    left_sequence = left_vlmc.generate_sequence(sequence_length, pre_sample_length)
    right_sequence = right_vlmc.generate_sequence(sequence_length, pre_sample_length)

    right_context_counters, right_transition_counters = self._count_events(right_vlmc, left_sequence)
    new_right_vlmc = self._create_vlmc_by_estimating_probabilities(
      self.alphabet, right_context_counters, right_transition_counters, right_vlmc.tree[""])

    # distance = self.d.distance(right_vlmc, new_right_vlmc)
    distance = self._perform_stats_test(new_right_vlmc, right_vlmc, right_context_counters)
    return distance

  cdef object _create_vlmc_by_estimating_probabilities(self, alphabet, context_counters, transition_counters,  acgt_content):
    cdef dict tree = {}
    for context, count in context_counters.items():
      if count == 0:
        tree[context] = {'A': 0, 'C': 0, 'G': 0, 'T': 0}
      else:
        tree[context] = {}
        for character in alphabet:
          tree[context][character] = transition_counters[context][character] / count

    return VLMC(tree, "estimated")

  cdef tuple _count_events(self, right_vlmc, sequence):
    cdef dict context_counters = {}
    cdef dict transition_counters = {}
    # Initialize counters
    for context in right_vlmc.tree.keys():
      context_counters[context] = 0
      transition_counters[context] = {}
      for character in self.alphabet:
        transition_counters[context][character] = 0

    sequence_so_far = ""
    current_contexts = []
    for character in sequence:
      for context in current_contexts:
        context_counters[context] += 1
        transition_counters[context][character] += 1

      sequence_so_far = sequence_so_far + character
      current_contexts = right_vlmc.get_all_contexts(sequence_so_far)

    return context_counters, transition_counters

  cdef double _perform_stats_test(self, estimated_vlmc, original_vlmc, context_count):
    expected_values = np.array([])
    observed_values = np.array([])
    for context in original_vlmc.tree.keys():
      times_visited_node = context_count[context]
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
