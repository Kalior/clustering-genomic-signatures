import math


cdef class NaiveParameterSampling(object):
  """
    Proposed by Levinson et al. for discrete-observation density hidden Markov models.
    Appears also in the paper by Juang et al. from 1985 on "A Probablistic Distance Measure For Hidden Markov Models".
  """

  def __init__(self):
    return

  cpdef double distance(self, left_vlmc, right_vlmc):
    # Assume this is the alphabet, only relevant case for us.
    cdef list alphabet = ['A', 'C', 'G', 'T']
    cdef dict left_tree = left_vlmc.tree, right_tree = right_vlmc.tree

    cdef double symmetric_distance = (self._assymmetric_distance(left_tree, right_tree, alphabet)
                          + self._assymmetric_distance(right_tree, left_tree, alphabet)) / 2
    return symmetric_distance

  cdef double _assymmetric_distance(self, left_tree, right_tree, alphabet):
    cdef double s
    s = sum(self._min_value_of_same_order(right_tree, k, left_tree[k][char_], char_)
            for k in left_tree.keys() for char_ in alphabet)

    s = s / (len(alphabet) * len(left_tree.keys()))
    s = math.sqrt(s)
    return s

  cdef double _min_value_of_same_order(self, tree, context, prob, char_):
    cdef list probabilities = [tree[k][char_] for k in tree.keys() if len(k) == len(context)] + [0]
    cdef double val = min(math.pow((val - prob), 2) for val in probabilities)
    return val
