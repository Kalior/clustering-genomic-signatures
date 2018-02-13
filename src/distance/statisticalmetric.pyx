import scipy.stats as stats
import numpy as np
from vlmc import VLMC


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
    p_values = np.arange(0, 0.001, 0.00005) # should come from a function
    for threshhold in p_values:
      left_vlmc = self.remove_unlikely_events(left_vlmc, threshhold)
      right_vlmc = self.remove_unlikely_events(right_vlmc, threshhold)
      if (not self.is_null_model(left_vlmc) and not self.is_null_model(right_vlmc)):
        # as long as none of the models were null-models, perform an equivalence test
        print("Performing equivalence test with p-value: " + str(threshhold))
        if self.equivalence_test(left_vlmc, right_vlmc):
          print("Found equality at p_value " + str(threshhold))
          return threshhold
    return 1


  cdef bint is_null_model(self, vlmc):
    # Not completely sure if this is everything that is neeed.
    if vlmc == None or not vlmc.tree: #if tree is empty
      return True
    for lookup_prob in vlmc.tree.values():
      if any(prob > 0 for prob in lookup_prob.values()):
        return False
    return True


  cdef object remove_unlikely_events(self, vlmc, threshhold_probability):
    stationary_distibution = vlmc.get_context_distribution()
    if stationary_distibution == None:
      # this should only happen when the _transition matrix_ is of size 0x0
      # i.e., when a vlmc has no contexts what so ever
      return VLMC({}, "null-vlmc")
    contexts_without_outgoing_transitions = []
    alphabet = ["A", "C", "G", "T"]
    for context in stationary_distibution.keys():
      nbr_transitions_to_keep = 0
      sum_probabilites_of_transitions_to_keep = 0
      for character in alphabet:
        event_probability = stationary_distibution[context] * vlmc.tree[context][character]
        if event_probability <= threshhold_probability:
          # if the event probability is smaller than threshhold, set its
          # corresponding transition probability to zero
          vlmc.tree[context][character] = 0
        else:
          # else keep it
          sum_probabilites_of_transitions_to_keep += vlmc.tree[context][character]
          nbr_transitions_to_keep += 1
      if nbr_transitions_to_keep == 0:
        # if no transitions left, delete context
        contexts_without_outgoing_transitions.append(context)

      for character in vlmc.tree[context]:
        # re-normalize the transitions probabilites that were kept
        if vlmc.tree[context][character] > 0:
          vlmc.tree[context][character] = vlmc.tree[context][character] / sum_probabilites_of_transitions_to_keep

    for context in contexts_without_outgoing_transitions:
      vlmc.tree.pop(context)

    return vlmc


  cdef bint equivalence_test(self, left_vlmc, right_vlmc):
    pre_sample_length = 500
    cdef str sequence = left_vlmc.generate_sequence(self.sequence_length, pre_sample_length)
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
      current_context = self.get_relvant_context(current_context, character, right_vlmc)

      if current_context == None:
        # this means that the vlmc could not have produced the next character given its context
        # exit with no equivalence for this starting context
        return False

    new_vlmc_tree = self.create_pst_by_estimating_probabilities(context_counters, transition_counters)
    chisq, p_value = self.perform_chi_squared_test(new_vlmc_tree, right_vlmc, context_counters)
    return 1 - p_value < self.significance_level


  cdef tuple perform_chi_squared_test(self, estimated_vlmc_tree, original_vlmc, context_count):
    expected_values = []
    observed_values = []
    for context in original_vlmc.tree.keys():
      times_visited_node = context_count[context]
      if times_visited_node > 0:
        alphabet = ["A", "C", "G", "T"]
        transition_probabilitites = list(map(lambda x: (x, original_vlmc.tree[context][x]), alphabet ))

        # find probabilites that are greater than zero
        probs_without_zeros = [item for item in transition_probabilitites if item[1] > 0]
        # loop through all of these exept one (last)
        for character_probability_tuple in probs_without_zeros[:-1]:
          character = character_probability_tuple[0]
          probability_original_vlmc = character_probability_tuple[1]
          probability_estimation = estimated_vlmc_tree[context][character]

          expected_frequency = times_visited_node * probability_original_vlmc
          observed_frequency = times_visited_node * probability_estimation
          expected_values.append(expected_frequency)
          observed_values.append(observed_frequency)
    chisq, p_value = stats.chisquare(f_obs=observed_values, f_exp=expected_values)
    return chisq, p_value


  cdef dict create_pst_by_estimating_probabilities(self, context_counters, transition_probabilities):
    tree = {}
    for context, count in context_counters.items():
      tree[context] = {}
      for character in ["A", "C", "G", "T"]:
        if count > 0:
          tree[context][character] = transition_probabilities[context][character] / count
        else:
          tree[context][character] = 0

    return tree


  cdef str get_relvant_context(self, context_before, next_char, vlmc_to_approximate):
    order = vlmc_to_approximate.order
    possible_contexts = vlmc_to_approximate.tree.keys()
    tmp_context = context_before + next_char
    # truncate to _order_ nbr of characters
    new_context = tmp_context[-order:]
    # Loop through to find the longest possible context applicable
    # TODO, extract this into to vlmc model
    for i in range(order+1):
      if not i == order:
        maybe_context = new_context[-(order-i):]
      else:
        maybe_context = ""
      if maybe_context in possible_contexts:
        return maybe_context
    # returns None if the approximated vlmc does not have any context
    # that fits the sequence
    return None


  cdef tuple initialize_counters(self, right_vlmc, start_context):
    context_counters = {}  # Map (Context -> Int)
    transition_counters = {}  # Map (Context -> Map (Character -> Int))
    current_context = start_context
    # Initialize counters
    for context in right_vlmc.tree.keys():
      context_counters[context] = 0
      transition_counters[context] = {}
      for character in ["A", "C", "G", "T"]:
        transition_counters[context][character] = 0
    return context_counters, transition_counters
