import math
import json
from queue import Queue
import os
from random import choices


cdef class VLMC(object):
  cdef public dict tree
  cdef public str name
  cdef public int order

  def __init__(self, tree, name):
    self.tree = tree
    self.name = name
    self.order = self._calculate_order(tree)

  def __str__(self):
    return self.name

  def __hash__(self):
    return hash(self.name)

  def __eq__(self, other):
    return self.name == other.name

  @classmethod
  def from_json(cls, s, name=""):
    """
      Expects the json to be in the following format:
      '{"":{"A":0.5,"B":0.5},"A":{"B":0.5,"A":0.5},"B":{"A":0.5,"B":0.5},"BA":{"A":0.5,"B":0.5},"AA":{"A":0.5,"B":0.5}}'
    """
    tree = json.loads(s)
    return VLMC(tree, name)

  @classmethod
  def from_json_dir(cls, directory):
    all_vlmcs = []
    for file in [f for f in os.listdir(directory) if f.endswith(".json")]:
      name, _ = os.path.splitext(file)
      name_without_parameter = VLMC.strip_parameters_from_name(name)
      with open(os.path.join(directory, file)) as f:
        tree = json.load(f)
        all_vlmcs.append(VLMC(tree, name_without_parameter))

    return all_vlmcs

  @classmethod
  def strip_parameters_from_name(cls, signature_name):
    stripped_order_prefix = signature_name[3:]
    split_name = stripped_order_prefix.split("_")
    removed_start_stop_indicies = split_name[:len(split_name) - 2]
    remove_empty_strings = [s for s in removed_start_stop_indicies if s != ""]
    aid = '_'.join(remove_empty_strings)
    return aid

  def to_json(self):
    """
      Returns the vlmc tree in the same format as from_json expects.
    """
    return json.dumps(self.tree)

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
        return 0
      log_likelihood += math.log(prob)
      sequence_so_far += s
    return log_likelihood


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


  cpdef str generate_sequence(self, sequence_length, pre_sample_length):
    total_length = sequence_length + pre_sample_length
    cdef str generated_sequence = ""
    for i in range(total_length):
      # only send the last /order/ number of characters to generate next letter
      next_letter = self._generate_next_letter(generated_sequence[-self.order:])
      generated_sequence += next_letter
    # return the suffix with length sequence_length
    return generated_sequence[-sequence_length:]

  cdef str _generate_next_letter(self, current_sequence):
    cdef list letters = ["A", "C", "G", "T"]
    probabilities = map(lambda char_: self._probability_of_char_given_sequence(
        char_, current_sequence), letters)
    return choices(letters, weights=probabilities)[0]

  def _calculate_order(self, tree):
    return max(map(lambda k: len(k), tree.keys()))

  def mirror(self):
    conversion_dict = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}

    mirror_tree = {}
    for context, probs in self.tree.items():
      mirror_context = "".join([conversion_dict[char_] for char_ in context])
      mirror_probs = {}
      for char_, prob in probs.items():
        mirror_probs[conversion_dict[char_]] = prob

      mirror_tree[mirror_context] = mirror_probs

    return VLMC(mirror_tree, "mirror_" + self.name)

if __name__ == "__main__":
  s = '{"":{"A":0.5,"B":0.5},"A":{"B":0.5,"A":0.5},"B":{"A":0.5,"B":0.5},"BA":{"A":0.5,"B":0.5},"AA":{"A":0.5,"B":0.5}}'

  vlmc = VLMC.from_json(s)
  print(str(vlmc))
  print(vlmc.to_json())

  print(vlmc.negative_log_likelihood("ABABBABA"))
  print([str(v) for v in VLMC.from_json_dir('../../trees')])
