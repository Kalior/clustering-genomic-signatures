class NegativeLogLikelihood():
  """
  Calculates the distance between two VLMCs.
  
  D(left, right) = (D(left, right) + D(right, left)) / 2
  Where, D(x, y) is the cross entropy calculated by generating a sequence S_x from model x,
  and computing:
  D(x, y) = log Pr(S_x | x) - log Pr(S_x | y)
  Where log Pr(s | x) is the negative log-likelihood of sequence s given vlmc x.
  """
  generated_sequence_length = -1
  def __init__(self, sequence_length):
    self.generated_sequence_length = sequence_length

  def distance(self, left_vlmc, right_vlmc):
    d_left_right = self._calculate_cross_entropy(left_vlmc, right_vlmc)
    d_right_left = self._calculate_cross_entropy(right_vlmc, left_vlmc)
    return (d_left_right + d_right_left)/2

  def _calculate_cross_entropy(self, left, right):
    generated_sequence = left.generate_sequence(self.generated_sequence_length)
    return (left.log_likelihood(generated_sequence)
            - right.log_likelihood(generated_sequence))

