import scipy.stats as stats
import numpy as np
from vlmc import VLMC, AbsorbingStateException

import copy

cdef class StatisticalMetric(object):
  cdef double significance_level
  cdef int sequence_length


  def __init__(self, sequence_length, significance_level):
      self.sequence_length = sequence_length
      self.significance_level = significance_level


  """
  1. Generate a sequence A of length N from G₁
  2. For every starting state in vₖ in G₂, estimate the transition
     probabilities if it were to generate the sequence
  3. For every starting state vₖ that accepted the sequence, generate
     G₂' from the estimated transition probabilites in 2)
  4. Determine Chi-squared statistic with significance level of α and
     nₜ - nₛ degrees of freedom.  Where nₜ = number of transitions and
     nₛ is the number of states in G₂.  This can be done with standard
     tables or calculating the value.
  5. Compute the Χ² statistic for comparing model G₂ with models G₂'ₖ as
     [see paper].
  6. If Χ²ₖ ≤ Χ²ₐ for any G₂'ₖ the Chi-squared test test accepts with
     significance a the hypothesis that the PDFs in G₂ are consistent
     with the PDFs in that G₂'ₖ. Since G₂'ₖ is calculated using a data
     trace from G₁ , this means that their statistics are
     consistent. Exit with equivalence.
  7. Else exit with no equivalence.
  """
  cpdef double distance(self, left_vlmc_original, right_vlmc_original):
    left_vlmc = copy.deepcopy(left_vlmc_original)
    right_vlmc = copy.deepcopy(right_vlmc_original)

    print("Measuring distance between {}\t and {}".format(left_vlmc.name, right_vlmc.name))
    p_values = np.arange(0, 0.001, 0.00001) # should come from a function
    for threshhold in p_values:
      left_vlmc = self._remove_unlikely_leaf_node_events(left_vlmc, threshhold)
      right_vlmc = self._remove_unlikely_leaf_node_events(right_vlmc, threshhold)
      if (not self.is_null_model(left_vlmc) and not self.is_null_model(right_vlmc)):
        # as long as none of the models were null-models, perform an equivalence test
        if self.equivalence_test(left_vlmc, right_vlmc):
          print("Found equality at p_value " + str(threshhold))
          return threshhold
    print("Fround no equality")
    return 1


  cdef bint is_null_model(self, vlmc):
    # Not completely sure if this is everything that is neeed.
    if vlmc == None or not vlmc.tree: #if tree is empty
      return True
    for lookup_prob in vlmc.tree.values():
      if any(prob > 0 for prob in lookup_prob.values()):
        return False
    return True


  cdef object _remove_unlikely_leaf_node_events(self, vlmc, threshhold_event_probability):
    has_removed_transition = True
    while has_removed_transition:
      has_removed_transition = self._remove_leaf_node_transitions(vlmc, threshhold_event_probability)
      contexts_to_delete = self._get_states_with_all_zero_probability_transitions(vlmc)
      self._delete_contexts(vlmc, contexts_to_delete)
      self._normalize_transition_probabilites(vlmc)
    return vlmc


  cdef bint _remove_leaf_node_transitions(self, vlmc, threshhold_event_probability):
    has_removed_transition = False
    stationary_distibution = vlmc.get_context_distribution()
    if stationary_distibution == None:
      # Happens if there are no contexts in the model
      return False
    for context in stationary_distibution.keys():
      if self._is_leaf_context(context, vlmc):
        for character in vlmc.alphabet:
          event_probability = stationary_distibution[context] * vlmc.tree[context][character]
          if event_probability <= threshhold_event_probability and vlmc.tree[context][character] > 0:
            # if the event probability is smaller than threshhold and
            # it has not been set to zero earlier, set its
            # corresponding transition probability to zero
            vlmc.tree[context][character] = 0
            has_removed_transition = True
    return has_removed_transition


  cdef bint _is_leaf_context(self, context, vlmc):
    possible_leaves = list(map(lambda c: context + c, vlmc.alphabet))
    # checks if any of the possible leaves does not exist as a key
    return any(map(lambda leaf: not leaf in vlmc.tree, possible_leaves))


  cdef _delete_contexts(self, vlmc, contexts):
    for context in contexts:
      vlmc.tree.pop(context)


  cdef list _get_states_with_all_zero_probability_transitions(self, vlmc):
    return list(filter(lambda context: sum(vlmc.tree[context].values()) == 0, vlmc.tree.keys()))


  cdef object _normalize_transition_probabilites(self, vlmc):
    for context in vlmc.tree.keys():
      total_weight = sum(vlmc.tree[context].values())
      # total weight should always be a positive number since absorbing states
      # has already been deleted
      for character, probabilty in vlmc.tree[context].items():
        # normalize probability
        vlmc.tree[context][character] = vlmc.tree[context][character] / total_weight


  cdef bint equivalence_test(self, left_vlmc, right_vlmc):
    pre_sample_length = 500
    cdef str sequence = ""
    try:
      sequence = left_vlmc.generate_sequence(self.sequence_length, pre_sample_length)
    except AbsorbingStateException:
      return False
    # For every starting state,
    possible_contexts = right_vlmc.tree.keys()
    for start_context in possible_contexts:
      if self.equality_test_given_starting_context(start_context, sequence, right_vlmc):
        return True
    return False


  cdef bint equality_test_given_starting_context(self, start_context, sequence, right_vlmc):
    current_context = start_context
    context_counters, transition_counters = self.initialize_counters(right_vlmc,                                                                        current_context)
    for character in sequence:
      context_counters[current_context] += 1
      transition_counters[current_context][character] += 1
      current_context = right_vlmc.get_context(current_context + character)

      if current_context == None:
        # this means that the vlmc could not have produced the next character given its context
        # exit with no equivalence for this starting context
        return False

    new_vlmc_tree = self.create_pst_by_estimating_probabilities(context_counters, transition_counters, right_vlmc.alphabet)
    chisq, p_value = self.perform_chi_squared_test(new_vlmc_tree, right_vlmc, context_counters)
    return 1 - p_value < self.significance_level



  cdef tuple perform_chi_squared_test(self, estimated_vlmc_tree, original_vlmc, context_count):
    expected_values = []
    observed_values = []
    for context in original_vlmc.tree.keys():
      times_visited_node = context_count[context]
      if times_visited_node > 0:
        transition_probabilitites = list(map(lambda x: (x, original_vlmc.tree[context][x]), original_vlmc.alphabet ))

        # find probabilites that are greater than zero
        probs_without_zeros = [(char_, prob) for (char_, prob) in transition_probabilitites if prob > 0]
        # loop through all of these exept one (last)
        for character, probability_original_vlmc in probs_without_zeros[:-1]:
          probability_estimation = estimated_vlmc_tree[context][character]

          expected_frequency = times_visited_node * probability_original_vlmc
          observed_frequency = times_visited_node * probability_estimation
          expected_values.append(expected_frequency)
          observed_values.append(observed_frequency)
    chisq, p_value = stats.chisquare(f_obs=observed_values, f_exp=expected_values)
    return chisq, p_value


  cdef dict create_pst_by_estimating_probabilities(self, context_counters, transition_probabilities, alphabet):
    tree = {}
    for context, count in context_counters.items():
      tree[context] = {}
      for character in alphabet:
        if count > 0:
          tree[context][character] = transition_probabilities[context][character] / count
        else:
          tree[context][character] = 0

    return tree


  cdef tuple initialize_counters(self, right_vlmc, start_context):
    context_counters = {}
    transition_counters = {}
    current_context = start_context
    # Initialize counters
    for context in right_vlmc.tree.keys():
      context_counters[context] = 0
      transition_counters[context] = {}
      for character in right_vlmc.alphabet:
        transition_counters[context][character] = 0
    return context_counters, transition_counters
