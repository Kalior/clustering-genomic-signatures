import numpy as np
import json
from queue import Queue
import os

class VLMC(object):
  """


  """
  graph = {}
  def __init__(self, graph):
    self.graph = graph

  def print(self):
    print(self.graph)

  @classmethod
  def from_json(cls, s):
    """
      Expects the json to be in the following format:
      '{"":{"A":0.5,"B":0.5},"A":{"B":0.5,"A":0.5},"B":{"A":0.5,"B":0.5},"BA":{"A":0.5,"B":0.5},"AA":{"A":0.5,"B":0.5}}'
    """
    graph = json.loads(s)
    return VLMC(graph)

  @classmethod
  def from_json_dir(cls, directory):
    all_vlmcs = []
    for file in [f for f in os.listdir(directory) if f.endswith(".json")]:
      with open(os.path.join(directory, file)) as f:
        data = json.load(f)
        all_vlmcs.append(VLMC(data))

    return all_vlmcs


  def to_json(self):
    """
      Returns the vlmc tree in the same format as from_json expects.
    """
    return json.dumps(self.graph)

  def negative_log_likelihood(self, sequence):
    sequence_so_far = ""
    log_likelihood = 0
    for s in sequence:
      prob = self.likelihood_of_char_given_sequence(s, sequence_so_far)
      log_likelihood += np.log(prob)
      sequence_so_far += s
    return -log_likelihood

  def likelihood_of_char_given_sequence(self, char, seq):
    reverse_seq = seq[::-1]
    depth = 0
    prob = 1
    current_node = reverse_seq[0:depth]
    while current_node in self.graph and depth < len(seq):
      prob = self.graph[current_node][char]
      depth += 1
      current_node = reverse_seq[0:depth]

    return prob


if __name__ == "__main__":
  s = '{"":{"A":0.5,"B":0.5},"A":{"B":0.5,"A":0.5},"B":{"A":0.5,"B":0.5},"BA":{"A":0.5,"B":0.5},"AA":{"A":0.5,"B":0.5}}'

  vlmc = VLMC.from_json(s)
  vlmc.print()
  print(vlmc.to_json())

  print(vlmc.negative_log_likelihood("ABABBABA"))
  print(VLMC.from_json_dir('../../trees'))
