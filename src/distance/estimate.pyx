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
    pre_sample_length = 500
    sequence_length = 10000

    left_sequence = left_vlmc.generate_sequence(sequence_length, pre_sample_length)

    context_counters, transition_counters = self._count_events(right_vlmc, left_sequence)
    new_right_vlmc = self._create_vlmc_by_estimating_probabilities(self.alphabet, context_counters, transition_counters, right_vlmc.tree[""])

    # distance = self.d.distance(right_vlmc, new_right_vlmc)
    distance = self._perform_stats_test(new_right_vlmc, right_vlmc, context_counters)
    return distance

  cdef object _create_vlmc_by_estimating_probabilities(self, alphabet, context_counters, transition_counters,  acgt_content):
    cdef dict tree = {}
    for context, count in context_counters.items():
      if count == 0:
        tree[context] = acgt_content
      else:
        tree[context] = {}
        for character in alphabet:
          tree[context][character] = transition_counters[context][character] / count

    tree = self._fix_empty_contexts(tree, acgt_content)

    return VLMC(tree, "estimated")

  cdef dict _fix_empty_contexts(self, tree, acgt_content):
    for context, prob in tree.items():
      if sum(p for p in prob.values()) == 0:
        tree[context] = acgt_content
    return tree

  cdef tuple _count_events(self, right_vlmc, sequence):
    cdef dict context_counters = {}
    cdef dict transition_counters = {}
    # Initialize counters
    for context in right_vlmc.tree.keys():
      context_counters[context] = 0
      transition_counters[context] = {}
      for character in self.alphabet:
        transition_counters[context][character] = 0

    current_context = ""
    for character in sequence:
      context_counters[current_context] += 1
      transition_counters[current_context][character] += 1
      current_context = right_vlmc.get_context(current_context + character)

    return context_counters, transition_counters

  cdef double _perform_stats_test(self, estimated_vlmc, original_vlmc, context_count):
    expected_values = np.array([p for i, p in enumerate(self._leaf_transitions(original_vlmc))])
    observed_values = np.array([p for i, p in enumerate(self._leaf_transitions(estimated_vlmc))])
    # expected_values = []
    # observed_values = []
    # for context in self._get_leaf_contexts(original_vlmc):
    #   times_visited_node = context_count[context]
    #   if times_visited_node > 0:
    #     # transition_probabilitites = [(x, original_vlmc.tree[context][x]) for x in self.alphabet]
    #     transition_probabilitites = self._leaf_transitions_in_context(original_vlmc, context)

    #     # find probabilites that are greater than zero
    #     probs_without_zeros = [(char_, prob) for (char_, prob) in transition_probabilitites if prob > 0]
    #     # loop through all of these exept one (last)
    #     for character, probability_original_vlmc in probs_without_zeros[:-1]:
    #       probability_estimation = estimated_vlmc.tree[context][character]

    #       expected_frequency = probability_original_vlmc
    #       observed_frequency = probability_estimation
    #       expected_values.append(expected_frequency)
    #       observed_values.append(observed_frequency)

    statistic, p_value = stats.power_divergence(f_obs=observed_values, f_exp=expected_values, lambda_="pearson")
    # observed_mean = observed_values.mean()
    # observed_var = observed_values.var()

    # expected_mean = expected_values.mean()
    # expected_var = expected_values.var()

    # distance = np.linalg.norm(expected_values - observed_values)
    distance = statistic
    return distance

  cdef list _leaf_transitions(self, vlmc):
    return [p for context in vlmc.tree.keys() for (_, p) in self._leaf_transitions_in_context(vlmc, context)]

  cdef list _leaf_transitions_in_context(self, vlmc, context):
    return [(c, vlmc.tree[context][c]) for c in self.alphabet if not (context + c) in vlmc.tree]

  cdef list _get_leaf_contexts(self, vlmc):
    return [c for c in vlmc.tree.keys() if self._is_leaf_context(c, vlmc)]

  cdef bint _is_leaf_context(self, context, vlmc):
    possible_leaves = [context + c for c in self.alphabet]
    # leaf contexts are defined as having no children at all
    return all(not leaf in vlmc.tree for leaf in possible_leaves)
