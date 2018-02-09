import scipy.stats as stats
import numpy as np

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
  cpdef double distance(self, left_vlmc, right_vlmc):
    for p_value in np.arange(0, 0.05, 0.001):
      left_vlmc.tree = self.remove_larger_probabilities(left_vlmc.tree, p_value)
      right_vlmc.tree = self.remove_larger_probabilities(right_vlmc.tree, p_value)
      if (not self.is_null_model(left_vlmc) and not self.is_null_model(right_vlmc)):
        if self.equivalence_test(left_vlmc, right_vlmc):
          print("Found equality at p_value " + str(p_value))
          return p_value
    return 1

  cdef bint is_null_model(self, vlmc):
    # TODO, this is clearly wrong
    return False
    for lookup_prob in vlmc.tree.values():
      if any(prob > 0 for prob in lookup_prob.values()):
        return False
    return True

  cdef dict remove_larger_probabilities(self, tree, p_value):
    contexts_to_remove = []
    for context in tree.keys():
      nbr_greater_probabilities = 0
      sum_of_other_probabilites = 0
      for letter in tree[context]:
        if tree[context][letter] <= p_value:
          tree[context][letter] = 0
        else:
          sum_of_other_probabilites += tree[context][letter]
          nbr_greater_probabilities += 1
      if nbr_greater_probabilities == 0:
        contexts_to_remove.append(context)
      for letter in tree[context]:
        if tree[context][letter] > p_value:
          tree[context][letter] = tree[context][letter] / sum_of_other_probabilites
    for context in contexts_to_remove:
      tree.pop(context)
    return tree

  cdef bint equivalence_test(self, left_vlmc, right_vlmc):
    pre_sample_length = 500
    cdef str sequence = left_vlmc.generate_sequence(self.sequence_length,
                                                    pre_sample_length)
    # For every starting state,
    possible_contexts = right_vlmc.tree.keys()
    for start_context in possible_contexts:
      if self.check_from_context(start_context, sequence, right_vlmc):
        return True
    return False

  cdef bint check_from_context(self, start_context, sequence, right_vlmc):
    current_context = start_context
    # Map (Context -> Int)
    # Map (Context -> Map (Letter -> Int))
    context_counters, context_probabilities = self.initialize_counters(right_vlmc,
                                                                       current_context)
    for char_ in sequence:
      context_counters[current_context] += 1
      context_probabilities[current_context][char_] += 1
      current_context = self.get_next_context(current_context, char_, right_vlmc)
      if current_context == "":
        return False

    new_vlmc_tree = self.create_pst_by_estimating_probabilities(context_counters, context_probabilities)
    expected_values, observed_values = self.get_expected_observed_values(new_vlmc_tree, right_vlmc, context_counters)
    chisq, p_value = stats.chisquare(f_obs=observed_values, f_exp=expected_values)
    if 1 - p_value < self.significance_level:
      return True

  cdef int nbr_of_non_zero_probabilities(self, vlmc):
    counter = 0
    for context, probabilites in vlmc.tree:
      for char_, prob in probabilites:
        if prob > 0:
          counter += 1
    return counter

  cdef tuple get_expected_observed_values(self, estimated_vlmc_tree, original_vlmc, context_count):
    expected_values = []
    observed_values = []
    for context in original_vlmc.tree.keys():
      times_visited_node = context_count[context]
      if times_visited_node > 0:
        alphabet = ["A", "C", "G", "T"]
        probabilites_original = list(zip(alphabet, list(map(lambda x: original_vlmc.tree[context][x], alphabet ))))
        # find probabilites that are greater than zero
        probs_without_zeros = [item for item in probabilites_original if item[1] > 0]
        # loop through all of these exept one (last)
        for char_prob in probs_without_zeros[:-1]:
          letter = char_prob[0]
          probability_original_vlmc = char_prob[1]
          probability_estimation = estimated_vlmc_tree[context][letter]

          expected_frequency = times_visited_node*probability_original_vlmc
          observed_frequency = times_visited_node*probability_estimation
          expected_values.append(expected_frequency)
          observed_values.append(observed_frequency)
    return expected_values, observed_values

  cdef dict create_pst_by_estimating_probabilities(self, context_counters, context_probabilites):
    tree = {}
    for context, count in context_counters.items():
      tree[context] = {}
      for char_ in ["A", "C", "G", "T"]:
        if count > 0:
          tree[context][char_] = context_probabilites[context][char_] / count
        else:
          tree[context][char_] = 0

    return tree

  cdef str get_next_context(self, context_before, next_char, vlmc_to_approximate):
    order = vlmc_to_approximate.order
    possible_contexts = vlmc_to_approximate.tree.keys()
    tmp_context = next_char + context_before
    # truncate to _order_ nbr of characters
    new_context = tmp_context[:order]
    # Loop through to find the longest possible context applicable
    for i in range(order):
      maybe_context = new_context[:order-i]
      if maybe_context in possible_contexts:
        return maybe_context
    return ""

  cdef tuple initialize_counters(self, right_vlmc, start_context):
    context_counters = {}  # Map (Context -> Int)
    context_probabilities = {}  # Map (Context -> Map (Letter -> Int))
    current_context = start_context
    # Initialize counters
    for context in right_vlmc.tree.keys():
      context_counters[context] = 0
      context_probabilities[context] = {}
      for char_ in ["A", "C", "G", "T"]:
        context_probabilities[context][char_] = 0
    return context_counters, context_probabilities
