import math
import json
from queue import Queue
import os
import random
import numpy as np
import scipy
from absorbing_state_exception import AbsorbingStateException


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
    if context == None:
      return 0
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
    return None



  cpdef str generate_sequence(self, sequence_length, pre_sample_length):
    total_length = sequence_length + pre_sample_length
    cdef str generated_sequence = ""

    if not "" in self.tree:
      # If the empty string does not exist in the vlmc, start
      # generating the sequence from a random context.
      # TODO: The probability of getting context c should be proportional
      # the stationary distribution of c.
      if len(self.tree) == 0:
        # if there are no contexts, we cant generate a sequence
        raise AbsorbingStateException
      generated_sequence += random.choice(list(self.tree.keys()))

    for i in range(total_length):
      # only send the last /order/ number of characters to generate next letter
      next_letter = self._generate_next_letter(generated_sequence[-self.order:])
      generated_sequence += next_letter
    # return the suffix with length sequence_length
    return generated_sequence[-sequence_length:]


  cdef str _generate_next_letter(self, current_sequence):
    probabilities = list(map(lambda c: self._probability_of_char_given_sequence(
      c, current_sequence), self.alphabet))

    # if none of the probabilities are larger than zero, raise exception
    if all(map(lambda x: x == 0, probabilities)):
      raise AbsorbingStateException("When generating a new letter")

    array = random.choices(self.alphabet, weights=probabilities)
    return array[0] # is list, take only element


  def _calculate_order(self, tree):
    if tree:
      return max(map(lambda k: len(k), tree.keys()))
    return 0


  def _get_transition_matrix(self):
    """Creates the transition matrix of the VLMC, where each row sums to one.

    The transition matrix is the square matrix A such that for each
    context c in the VLMC, there exists a row in A, that contains the
    probability of going from c to any other context c' (the columns).
    Since the alphabet is only 4 letters long, there will only be up
    to 4 non-zero probabilities per row.

    """
    nbr_of_states = len(self.tree.keys())
    rows = np.zeros((nbr_of_states, nbr_of_states), np.float32)
    for row, from_context in enumerate(self.tree.keys()):
      reachable_contexts = self._get_reachable_states(from_context)

      for column, to_context in enumerate(self.tree.keys()):
        prob = 0
        # if to_context is reachable, set probability accordingly
        for state, probability in reachable_contexts:
          if state == to_context:
            prob = probability
        rows[row, column] = prob

    return rows


  cpdef list _get_reachable_states(self, from_context):
    reachable_contexts = []
    for character in self.alphabet:
      probability_of_char = self.tree[from_context][character]
      if probability_of_char > 0:
        to_context = self.get_context(from_context + character)
        reachable_contexts.append((to_context, probability_of_char))
    return reachable_contexts


  cpdef dict get_context_distribution(self):
    """ Returns the stationary distribution probabilites of the contexts.

    If matrix A is the transition matrix and x is the current state
    then transpose(A)*x is the next state.  This function returns the
    vector v such that transpose(A)*v = v, i.e., it returns the
    stationary distribution of this vlmc """
    transition_matrix = self._get_transition_matrix()
    if transition_matrix.size == 0:
      return None
    matrix = np.transpose(transition_matrix)

    # for debugging
    # np.set_printoptions(threshold=np.NaN)

    try:
      # Get the eigen vector with eigen value 1 if it exist
      values, vectors = scipy.sparse.linalg.eigs(matrix, k=1, sigma=1)
    except scipy.sparse.linalg.eigen.arpack.ArpackNoConvergence:
      print("Calculation of eigenvector did not converge, using estimated stationary distribution.")
      return self._estimated_context_distribution(50000)

    eigen_vector = vectors[:, 0]

    # normalize the probabilites, so they sum to 1:
    probability_weight = np.sum(eigen_vector)
    normalized_eigen_vector = (np.real(eigen_vector) / probability_weight)

    # fill distribution dictionary
    stationary_distibution = {}
    for i, context in enumerate(self.tree.keys()):
      stationary_distibution[context] = normalized_eigen_vector[i]
    return stationary_distibution


  cpdef dict _estimated_context_distribution(self, sequence_length):
    sequence = ""
    try:
      sequence = self.generate_sequence(sequence_length, 500)
    except AbsorbingStateException:
      return {}
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


if __name__ == "__main__":
  s = '{"":{"A":0.5,"B":0.5},"A":{"B":0.5,"A":0.5},"B":{"A":0.5,"B":0.5},"BA":{"A":0.5,"B":0.5},"AA":{"A":0.5,"B":0.5}}'

  vlmc = VLMC.from_json(s)
  print(str(vlmc))
  print(vlmc.to_json())

  print(vlmc.negative_log_likelihood("ABABBABA"))
  print([str(v) for v in VLMC.from_json_dir('../../trees')])
