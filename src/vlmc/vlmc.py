import numpy as np
import json
from queue import Queue
import os
from random import choices

class VLMC(object):
  """


  """
  graph = {}
  name = ""
  order = -1
  def __init__(self, graph, name):
    self.graph = graph
    self.name = name
    self.order = self._calculate_order(graph)

  def __str__(self):
    return self.name

  @classmethod
  def from_json(cls, s, name=""):
    """
      Expects the json to be in the following format:
      '{"":{"A":0.5,"B":0.5},"A":{"B":0.5,"A":0.5},"B":{"A":0.5,"B":0.5},"BA":{"A":0.5,"B":0.5},"AA":{"A":0.5,"B":0.5}}'
    """
    graph = json.loads(s)
    return VLMC(graph, name)

  @classmethod
  def from_json_dir(cls, directory):
    all_vlmcs = []
    for file in [f for f in os.listdir(directory) if f.endswith(".json")]:
      name, _ = os.path.splitext(file)
      with open(os.path.join(directory, file)) as f:
        graph = json.load(f)
        all_vlmcs.append(VLMC(graph, name))

    return all_vlmcs


  def to_json(self):
    """
      Returns the vlmc tree in the same format as from_json expects.
    """
    return json.dumps(self.graph)

  def log_likelihood(self, sequence):
    sequence_so_far = ""
    log_likelihood = 0
    for s in sequence:
      prob = self._probability_of_char_given_sequence(s, sequence_so_far)
      log_likelihood += np.log(prob)
      sequence_so_far += s
    return log_likelihood

  def _probability_of_char_given_sequence(self, char, seq):
    reverse_seq = seq[::-1]
    depth = 0
    prob = 1
    current_node = reverse_seq[0:depth]
    while current_node in self.graph and depth < len(seq):
      prob = self.graph[current_node][char]
      depth += 1
      current_node = reverse_seq[0:depth]
    return prob

  def generate_sequence(self, sequence_length):
    generated_sequence = ""
    for i in range(sequence_length):
      generated_sequence += self._generate_next_letter(generated_sequence)
    return generated_sequence

  def _generate_next_letter(self, current_sequence):
    letters = ["A", "C", "G", "T"]
    probabilities = map(lambda char: self._probability_of_char_given_sequence(char, current_sequence), letters)
    return choices(letters, weights = probabilities)[0]

  def _calculate_order(self, graph):
    return max(map(lambda k: len(k), graph.keys()))


if __name__ == "__main__":
  s = '{"":{"A":0.5,"B":0.5},"A":{"B":0.5,"A":0.5},"B":{"A":0.5,"B":0.5},"BA":{"A":0.5,"B":0.5},"AA":{"A":0.5,"B":0.5}}'

  vlmc = VLMC.from_json(s)
  print(str(vlmc))
  print(vlmc.to_json())

  print(vlmc.negative_log_likelihood("ABABBABA"))
  print([str(v) for v in VLMC.from_json_dir('../../trees')])
