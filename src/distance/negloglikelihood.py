sequence_length = 2000

def distance(left_vlmc, right_vlmc):
  """
  Calculates the distance between two VLMCs.

  D(left, right) = (D(left, right) + D(right, left)) / 2
  Where, D(x, y) is the cross entropy calculated by generating a sequence S_x from model x,
  and computing:
  D(x, y) = log Pr(S_x | x) - log Pr(S_x | y)
  Where log Pr(s | x) is the negative log-likelihood of sequence s given vlmc x.
  """

  d_left_right = calculate_cross_entropy(left_vlmc, right_vlmc, sequence_length)
  d_right_left = calculate_cross_entropy(right_vlmc, left_vlmc, sequence_length)
  return (d_left_right + d_right_left)/2


def calculate_cross_entropy(left, right, sequence_length):
  generated_sequence = left.generate_sequence(sequence_length)
  return (-left.negative_log_likelihood(generated_sequence)
          + right.negative_log_likelihood(generated_sequence))

