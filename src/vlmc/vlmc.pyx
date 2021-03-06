import math
import json
from queue import Queue
import os
from random import choices


cdef class VLMC(object):
  cdef public dict tree
  cdef public str name
  cdef public int order
  cdef public str sequence
  cdef public list alphabet
  cdef dict occurrence_probabilites

  def __init__(self, tree, name, occurrence_probability):
    self.tree = tree
    self.name = name
    self.order = self._calculate_order(tree)
    self.sequence = ""
    self.alphabet = ['A', 'C', 'G', 'T']
    self.occurrence_probabilites = occurrence_probability

  def __str__(self):
    return self.name

  def __hash__(self):
    return hash(self.name)

  def __eq__(self, other):
    return other is not None and self.name == other.name

  @classmethod
  def from_json(cls, s, name=""):
    """
      Expects the json to be in the following format:
      '{"":{"A":0.5,"B":0.5},"A":{"B":0.5,"A":0.5},"B":{"A":0.5,"B":0.5},"BA":{"A":0.5,"B":0.5},"AA":{"A":0.5,"B":0.5}}'
    """
    vlmc_info = json.loads(s)
    if "tree" in vlmc_info and "occurrence_probability" in vlmc_info:
      tree = vlmc_info["tree"]
      occurrences = vlmc_info["occurrence_probability"]
    else: # assume its the old format where the json represented the tree
      tree = vlmc_info
      occurrences = {} # let it crash if we accidantly use old
    return VLMC(tree, name, occurrences)

  @classmethod
  def from_json_dir(cls, directory):
    all_vlmcs = []
    for file in [f for f in os.listdir(directory) if f.endswith(".json")]:
      name, _ = os.path.splitext(file)
      name_without_parameter = VLMC.strip_parameters_from_name(name)
      with open(os.path.join(directory, file)) as f:
        vlmc_dict = json.load(f)
        tree = vlmc_dict["tree"]
        occurrence_probability = vlmc_dict["occurrence_probability"]
        all_vlmcs.append(VLMC(tree, name_without_parameter, occurrence_probability))

    return all_vlmcs

  @classmethod
  def strip_parameters_from_name(cls, signature_name):
    split_name = signature_name.split("_")
    # pst_vlmc_name_23_23 -> pst, vlmc, name, 23, 23
    removed_start_stop_indicies = split_name[1:len(split_name) - 2]
    # pst, vlmc, name, 23, 23 -> vlmc, name
    remove_empty_strings = [s for s in removed_start_stop_indicies if s != ""]
    aid = '_'.join(remove_empty_strings) # vlmc, name -> vlmc_name
    return aid

  def to_json(self):
    """
      Returns the vlmc tree in the same format as from_json expects.
    """
    as_dict = {"tree": self.tree, "occurrence_probability": self.occurrence_probabilites}
    return json.dumps(as_dict)

  cpdef double log_likelihood_ignore_initial_bias(self, sequence):
    # skip the first /order/ characters to ignore the bias from
    # where the sequence was cut/taken
    return self._log_likelihood(sequence, self.order)

  cpdef double log_likelihood(self, sequence):
    return self._log_likelihood(sequence, 0)

  cdef double _log_likelihood(self, sequence, nbr_skipped_letters):
    # assume we already looked at the first nbr_skipped_letters
    cdef str sequence_so_far = sequence[:nbr_skipped_letters]
    cdef str sequence_left = sequence[nbr_skipped_letters:]
    cdef double log_likelihood = 0.0
    for s in sequence_left:
      prob = self._probability_of_char_given_sequence(s, sequence_so_far[-self.order:])
      if prob == 0:
        # means the vlmc could not possibly have generated the sequence
        # corresponds to prob == e^-1000
        log_likelihood -= 1000
      else:
        log_likelihood += math.log(prob)
      sequence_so_far += s
    return log_likelihood

  cpdef double likelihood(self, sequence):
    return self._likelihood(sequence, 0)
  
  cdef double _likelihood(self, sequence, nbr_skipped_letters):
    # assume we already looked at the first nbr_skipped_letters
    cdef str sequence_so_far = sequence[:nbr_skipped_letters]
    cdef str sequence_left = sequence[nbr_skipped_letters:]
    cdef double likelihood = 1.0
    cdef double prob = -1
    for s in sequence_left:
      prob = self._probability_of_char_given_sequence(s, sequence_so_far[-self.order:])
      if prob == 0:
        return 0.0
      else:
        likelihood *= prob
      sequence_so_far += s
    return likelihood
  

  cdef double _probability_of_char_given_sequence(self, character, seq):
    if len(seq) == self.order and seq in self.tree:
      # early return if possible. will be often if the model is a full Markov chain
      return self.tree[seq][character]

    cdef str context = self.get_context(seq)
    return self.tree[context][character]


  cpdef str get_context(self, sequence):
    if len(sequence) <= self.order and sequence in self.tree:
      return sequence
    for i in range(self.order+1):
      if not i == self.order:
        maybe_context = sequence[-(self.order-i):]
      else:
        maybe_context = ""
      if maybe_context in self.tree:
        return maybe_context
    raise RuntimeError("get_context vlmc.pyx")

  cpdef list get_all_contexts(self, sequence):
    possible_contexts = []
    max_possible_context_length = min(len(sequence), self.order)
    for i in range(max_possible_context_length + 1):
      if not i == max_possible_context_length:
        maybe_context = sequence[-(max_possible_context_length-i):]
      else:
        maybe_context = ""
      if maybe_context in self.tree:
        possible_contexts.append(maybe_context)

    return possible_contexts

  cpdef str generate_sequence(self, sequence_length, pre_sample_length):
    if len(self.sequence) == sequence_length:
      return self.sequence

    total_length = sequence_length + pre_sample_length
    cdef str generated_sequence = self.generate_sequence_from(total_length, "")
    self.sequence = generated_sequence[-sequence_length:]
    return generated_sequence[-sequence_length:]

  cpdef str generate_sequence_from(self, sequence_length, context):
    generated_sequence = context
    for i in range(sequence_length):
      # only send the last /order/ number of characters to generate next letter
      next_letter = self._generate_next_letter(generated_sequence[-self.order:])
      generated_sequence += next_letter
    # return the suffix with length sequence_length
    return generated_sequence[len(context):]

  cdef str _generate_next_letter(self, current_sequence):
    probabilities = map(lambda char_: self._probability_of_char_given_sequence(
        char_, current_sequence), self.alphabet)
    return choices(self.alphabet, weights=probabilities)[0]

  def _calculate_order(self, tree):
    return max(map(lambda k: len(k), tree.keys()))

  cpdef dict estimated_context_distribution(self, sequence_length):
    sequence = self.generate_sequence(sequence_length, 500)
    context_counters = self._count_state_occourances(sequence)
    context_distribution = {}

    for context, frequency in context_counters.items():
      prob_of_state = frequency / sequence_length
      context_distribution[context] = prob_of_state

    return context_distribution


  cdef dict _count_state_occourances(self, sequence):
    state_count = {}

    for i in range(len(sequence) - self.order):
      current_sequence = sequence[i:i + self.order]
      matching_state = self.get_context(current_sequence)
      if matching_state in state_count:
        state_count[matching_state] += 1
      else:
        # give initial value of 1
        state_count[matching_state] = 1

    return state_count

  cpdef void reset_sequence(self):
    self.sequence = ""


  cpdef double occurrence_probability(self, state):
    if len(self.occurrence_probabilites) == 0:
      raise RuntimeError("No information about occurrence probabilites")

    return self.occurrence_probabilites[state]


if __name__ == "__main__":
  s = '{"":{"A":0.5,"B":0.5},"A":{"B":0.5,"A":0.5},"B":{"A":0.5,"B":0.5},"BA":{"A":0.5,"B":0.5},"AA":{"A":0.5,"B":0.5}}'

  vlmc = VLMC.from_json(s)
  print(str(vlmc))
  print(vlmc.to_json())

  print(vlmc.negative_log_likelihood("ABABBABA"))
  print([str(v) for v in VLMC.from_json_dir('../../trees')])
