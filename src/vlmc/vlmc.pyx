import math
import json
from queue import Queue
import os
from random import choices


cdef class VLMC(object):
  cdef public dict tree
  cdef public str name
  cdef int order

  def __init__(self, tree, name):
    self.tree = tree
    self.name = name
    self.order = self._calculate_order(tree)

  def __str__(self):
    return self.name

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
      with open(os.path.join(directory, file)) as f:
        tree = json.load(f)
        all_vlmcs.append(VLMC(tree, name))

    return all_vlmcs

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
      log_likelihood += math.log(prob)
      sequence_so_far += s
    return log_likelihood

  cdef double _probability_of_char_given_sequence(self, char, seq):
    cdef str reverse_seq = seq[::-1]
    if len(seq) == self.order and reverse_seq in self.tree:
      # early return if possible. will be often if the model is a full Markov chain
      return self.tree[reverse_seq][char]
    cdef int depth = 0
    cdef double prob = 1.0
    cdef str current_node = ""
    # Search down the tree for either the order of the vlmc, or the length of the string
    while current_node in self.tree and depth < len(seq):
      prob = self.tree[current_node][char]
      depth += 1
      current_node = reverse_seq[0:depth]
    return prob

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


if __name__ == "__main__":
  s = '{"":{"A":0.5,"B":0.5},"A":{"B":0.5,"A":0.5},"B":{"A":0.5,"B":0.5},"BA":{"A":0.5,"B":0.5},"AA":{"A":0.5,"B":0.5}}'

  vlmc = VLMC.from_json(s)
  print(str(vlmc))
  print(vlmc.to_json())

  print(vlmc.negative_log_likelihood("ABABBABA"))
  print([str(v) for v in VLMC.from_json_dir('../../trees')])
