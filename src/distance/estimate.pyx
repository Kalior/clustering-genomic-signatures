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
    sequence_length = 20000

    left_sequence = left_vlmc.generate_sequence(sequence_length, pre_sample_length)

    right_context_counters, right_transition_counters = self._count_events(
        right_vlmc, left_sequence, "")

    new_right_vlmc = self._create_vlmc_by_estimating_probabilities(
        right_context_counters, right_transition_counters, self.alphabet, right_vlmc.tree[""])

    distance = self.d.distance(right_vlmc, new_right_vlmc)
    return distance

  cdef object _create_vlmc_by_estimating_probabilities(self, context_counters, transition_probabilities, alphabet, acgt_content):
    cdef dict tree = {}
    for context, count in context_counters.items():
      if count == 0:
        tree[context] = acgt_content
      else:
        tree[context] = {}
        for character in alphabet:
          tree[context][character] = transition_probabilities[context][character] / count

    for context, prob in tree.items():
      if sum(p for p in prob.values()) == 0:
        tree[context] = acgt_content

    return VLMC(tree, "estimated")

  cdef tuple _count_events(self, right_vlmc, sequence, start_context):
    cdef dict context_counters = {}
    cdef dict transition_counters = {}
    # Initialize counters
    for context in right_vlmc.tree.keys():
      context_counters[context] = 0
      transition_counters[context] = {}
      for character in self.alphabet:
        transition_counters[context][character] = 0

    current_context = start_context
    for character in sequence:
      context_counters[current_context] += 1
      transition_counters[current_context][character] += 1
      current_context = right_vlmc.get_context(current_context + character)

    return context_counters, transition_counters
