import numpy as np

cdef class StepWiseNegativeLogLikelihood(object):
  """
    Calculates a step-wise version of negative log-likelihood between two models.

    Uses negative log-likelihood (NLL) as an equality check, and if two models are inequal,
    add a threshold to the generation and calculation of NLL which removes some probabilities.

    When equality is reached, return the threshold at that point.

    The value of negative log-likelihood which determines equality is a user parameter.
  """

  cdef double nll_threshold
  cdef int generated_sequence_length
  cdef int length_of_pregenerated_sequence

  def __init__(self, sequence_length=1000, length_of_pregenerated_sequence=500, nll_threshold=0.05):
    self.nll_threshold = nll_threshold
    self.generated_sequence_length = sequence_length
    self.length_of_pregenerated_sequence = length_of_pregenerated_sequence

  cpdef double distance(self, left_vlmc, right_vlmc):
    thresholds = np.arange(0, 1, 0.05)
    for threshold in thresholds:
      if self._symmetric_cross_entropy(left_vlmc, right_vlmc, threshold) < self.nll_threshold:
        return threshold

    return 1

  cdef double _symmetric_cross_entropy(self, left_vlmc, right_vlmc, threshold):
    cdef double d_left_right = self._calculate_cross_entropy(left_vlmc, right_vlmc, threshold)
    cdef double d_right_left = self._calculate_cross_entropy(right_vlmc, left_vlmc, threshold)
    return (d_left_right + d_right_left) / 2

  cdef double _calculate_cross_entropy(self, left_vlmc, right_vlmc, threshold):
    cdef str generated_sequence = left_vlmc.generate_sequence(self.generated_sequence_length,
                                                         self.length_of_pregenerated_sequence, threshold)
    return (left_vlmc.log_likelihood_ignore_initial_bias(generated_sequence, threshold)
            - right_vlmc.log_likelihood_ignore_initial_bias(generated_sequence, threshold)) / self.generated_sequence_length
