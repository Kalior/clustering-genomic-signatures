import numpy as np
import json
from queue import Queue


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
      Expects the json to be in a tree-like format:
      '{"label":"","children":[{"prob":0.5,"label":"A","children":[{"prob":0.5,"label":"B","children":[{"label":"A","prob":0.5},{"label":"B","prob":0.5}]},{"prob":0.5,"label":"A","children":[{"label":"A","prob":0.5},{"label":"B","prob":0.5}]}]},{"prob":0.5,"label":"B","children":[{"label":"A","prob":0.5},{"label":"B","prob":0.5}]}]}'
    """
    graph = {}

    data = json.loads(s)

    queue = Queue()
    queue.put({'data': data, 'prev_label': ""})

    while not queue.empty():
      item = queue.get()
      children = item['data']['children']

      key = item['data']['label'] + item['prev_label']
      #node = [{'label': child['label']: 'prob': child['prob']} for child in children]
      #graph[key] = node
      dict_children = {}
      for child in children:
        dict_children[child['label']] = child['prob']
      graph[key] = dict_children

      # Filter children who aren't nodes (have no own children)
      node_children = list(filter(lambda c: 'children' in c, children))
      [queue.put({'data': child, 'prev_label': key}) for child in node_children]

    return VLMC(graph)

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


s = '{"label":"","children":[{"prob":0.5,"label":"A","children":[{"prob":0.5,"label":"B","children":[{"label":"A","prob":0.5},{"label":"B","prob":0.5}]},{"prob":0.5,"label":"A","children":[{"label":"A","prob":0.5},{"label":"B","prob":0.5}]}]},{"prob":0.5,"label":"B","children":[{"label":"A","prob":0.5},{"label":"B","prob":0.5}]}]}'

vlmc = VLMC.from_json(s)
vlmc.print()
print(vlmc.to_json())

print(vlmc.negative_log_likelihood("ABABBABA"))
