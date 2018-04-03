import numpy as np
cimport numpy as np
FLOATTYPE = np.float32
ctypedef np.float32_t FLOATTYPE_t
from itertools import product, repeat

cdef class FixedLengthSequenceKLDivergence(object):
  """
  Calculates the Kullback-liebler divergence between two vlmcs given a sequence length
  D_kl (P || Q) := Σᵢ P(i)·log[ P(i)/Q(i) ]
  """
  cdef int fixed_length
  cdef list alphabet
  
  def __cinit__(self, string_length):
    self.fixed_length = string_length
    self.alphabet = ["A", "C", "G", "T"]

  cpdef double distance(self, left_vlmc, right_vlmc):
    # D_kl (P || Q) := Σᵢ P(i)·log[ P(i)/Q(i) ]
    cdef double KL_divergence = 0
    cdef list all_possible_sequences = [''.join(p) for p in product(self.alphabet, repeat=self.fixed_length)]
    cdef double p_i = -1
    cdef double q_i = -1
    cdef double contribution_i = -1
    
    for sequence in all_possible_sequences:
      p_i = left_vlmc.likelihood(sequence)
      if p_i == 0:
        contribution_i = 0
      else:
        if q_i == 0:
          raise RuntimeError("When calculating Kullback")
        q_i = right_vlmc.likelihood(sequence)
        contribution_i = p_i * np.log(p_i/q_i)
      KL_divergence += contribution_i
    return KL_divergence

