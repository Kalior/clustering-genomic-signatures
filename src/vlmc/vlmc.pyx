import math
import json
from queue import Queue
import os
from random import choices
import numpy as np
import scipy


cdef class VLMC(object):
  cdef public dict tree
  cdef public str name
  cdef public int order
  cdef public list alphabet


  def __init__(self, tree, name):
    self.tree = tree
    self.name = name
    self.order = self._calculate_order(tree)
    self.alphabet = ["A", "C", "G", "T"]


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
    aid = '_'.join(removed_start_stop_indicies)
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
    probabilities = map(lambda c: self._probability_of_char_given_sequence(
      c, current_sequence), self.alphabet)
    return random.choices(letters, weights=probabilities)[0] # is list, take only element


  def _calculate_order(self, tree):
    return max(map(lambda k: len(k), tree.keys()))

  def _get_transition_matrix(self):
    nbr_of_states = len(self.tree.keys())
    alphabet = ["A", "C", "G", "T"]
    rows = []
    for left in self.tree.keys():
      row = []
      contexts_we_can_get_to = []
      for char_ in alphabet:
        probability_of_char = self.tree[left][char_]
        if probability_of_char > 0:
          new_context = left + char_
          for i in range(self.order+1):
            truncated_context = new_context[-(self.order - i):]
            if truncated_context in self.tree:
              contexts_we_can_get_to.append((truncated_context, probability_of_char))
              break
      print(contexts_we_can_get_to)
      for right in self.tree.keys():
        prob = 0
        for x in contexts_we_can_get_to:
          if x[0] == right:
            prob = x[1]
        row.append(prob)
      rows.append(row)
      return np.array(rows)

  def get_context_distribution(self):
    # TODO, cleanup
    # if matrix A is the transition probabilites from row to column
    # and x is the current state then transpose(A)*x is the next state
    # this function returns the vector v such that transpose(A)*v = v,
    # i.e., it returns the stationary distribution of this vlmc
    transition_matrix = self._get_transition_matrix()
    matrix = np.transpose(transition_matrix)
    np.set_printoptions(threshold='nan')
    # Get one eigen vector, with eigen value 1
    # TODO, what happens if no eigen value 1 vector exists?
    values, vectors = scipy.sparse.linalg.eigs(matrix, k=1, sigma=1)
    np.set_printoptions(threshold=np.NaN)
    print(values[0])
    print(self.tree.keys())

    one = vectors[:, 0]
    sum_ = np.sum(one)
    scaled_vector = np.real(np.around(one.real / sum_, decimals=4))
    stationary_distibution = {}
    for i, context in enumerate(self.tree.keys()):
      # print("Probability of " + context + "\t" + str(scaled_vector[i]))
      stationary_distibution[context] = scaled_vector[i]
    # print(self.tree["GTA"])
    return stationary_distibution

if __name__ == "__main__":
  s = '{"":{"A":0.5,"B":0.5},"A":{"B":0.5,"A":0.5},"B":{"A":0.5,"B":0.5},"BA":{"A":0.5,"B":0.5},"AA":{"A":0.5,"B":0.5}}'

  vlmc = VLMC.from_json(s)
  print(str(vlmc))
  print(vlmc.to_json())

  print(vlmc.negative_log_likelihood("ABABBABA"))
  print([str(v) for v in VLMC.from_json_dir('../../trees')])
