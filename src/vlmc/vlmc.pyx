import math
import json
from queue import Queue
import os
from random import choices
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt


cdef class VLMC(object):
  cdef public dict tree
  cdef public str name
  cdef public int order
  cdef public str sequence
  cdef public list alphabet

  def __init__(self, tree, name):
    self.tree = tree
    self.name = name
    self.order = self._calculate_order(tree)
    self.sequence = ""
    self.alphabet = ['A', 'C', 'G', 'T']

  def __str__(self):
    return self.name

  def __hash__(self):
    return hash(self.name)

  def __eq__(self, other):
    return other is not None and self.name == other.name

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
    remove_empty_strings = [s for s in removed_start_stop_indicies if s != ""]
    aid = '_'.join(remove_empty_strings)
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
        # corresponds to prob == e^-1000
        log_likelihood -= 1000
      else:
        log_likelihood += math.log(prob)
      sequence_so_far += s
    return log_likelihood


  cdef double _probability_of_char_given_sequence(self, character, seq):
    if len(seq) == self.order and seq in self.tree:
      # early return if possible. will be often if the model is a full Markov chain
      return self.tree[seq][character]

    cdef str context = self.get_context(seq)
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
    raise RuntimeError("get_context vlmc.pyx")

  cpdef list get_all_contexts(self, sequence):
    possible_contexts = []
    max_possible_context_length = min(len(sequence), self.order)
    for i in range(max_possible_context_length + 1):
      if not i == max_possible_context_length:
        maybe_context = sequence[-(max_possible_context_length-i):]
      else:
        maybe_context = ""
      if maybe_context in self.tree:
        possible_contexts.append(maybe_context)

    return possible_contexts

  cpdef str generate_sequence(self, sequence_length, pre_sample_length):
    if len(self.sequence) == sequence_length:
      return self.sequence

    total_length = sequence_length + pre_sample_length
    cdef str generated_sequence = ""
    for i in range(total_length):
      # only send the last /order/ number of characters to generate next letter
      next_letter = self._generate_next_letter(generated_sequence[-self.order:])
      generated_sequence += next_letter
    # return the suffix with length sequence_length

    self.sequence = generated_sequence[-sequence_length:]
    return generated_sequence[-sequence_length:]

  cdef str _generate_next_letter(self, current_sequence):
    cdef list letters = ["A", "C", "G", "T"]
    probabilities = map(lambda char_: self._probability_of_char_given_sequence(
        char_, current_sequence), letters)
    return choices(letters, weights=probabilities)[0]

  def _calculate_order(self, tree):
    return max(map(lambda k: len(k), tree.keys()))

  cpdef dict estimated_context_distribution(self, sequence_length):
    sequence = self.generate_sequence(sequence_length, 500)
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

  def draw(self, metadata):
    G = nx.DiGraph()
    if self.name in metadata:
      root_name = metadata[self.name]['species']
    else:
      root_name = self.name
    G.add_node(root_name, inner=True)
    self._add_children(G, "", root_name)

    edges = G.edges(data='inner')
    inner_edges = [e for e in edges if e[2]]
    outer_edges = [e for e in edges if not e[2]]

    nodes = G.nodes(data='inner')
    inner_nodes = {n[0]: n[0] for n in nodes if n[1]}

    pos = graphviz_layout(G, prog='dot')

    nx.draw_networkx_nodes(G, pos, node_size=10, node_color='w')
    nx.draw_networkx_labels(G, pos, font_size=16, labels=inner_nodes)

    nx.draw_networkx_edges(G, pos, edgelist=inner_edges, arrows=True, edge_color='#ff7f00',
      width=2, style='solid')

    nx.draw_networkx_edges(G, pos, edgelist=outer_edges, arrows=False, edge_color='#007fff',
      width=1, style='dashed')

    edge_attributes=['symbol', 'prob']
    attributes = [nx.get_edge_attributes(G, attr) for attr in edge_attributes]
    keys = set([k for attrs in attributes for k in attrs.keys()])
    labels = {k: " ".join([attrs[k] for attrs in attributes if k in attrs]) for k in keys}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_size=16)

  def _add_children(self, G, context, root_name):
    for c in self.alphabet:
      child = c + context
      parent_label = context
      if parent_label == "":
        parent_label = root_name
      if child in self.tree:
        G.add_node(child, inner=True)
        G.add_edge(parent_label, child, symbol=c, prob="{:.2f}".format(self.tree[context][c]), inner=True)
        self._add_children(G, child, root_name)
      else:
        G.add_node(child, inner=False)
        G.add_edge(parent_label, child, symbol=c, prob="{:.2f}".format(self.tree[context][c]), inner=False)

if __name__ == "__main__":
  s = '{"":{"A":0.5,"B":0.5},"A":{"B":0.5,"A":0.5},"B":{"A":0.5,"B":0.5},"BA":{"A":0.5,"B":0.5},"AA":{"A":0.5,"B":0.5}}'

  vlmc = VLMC.from_json(s)
  print(str(vlmc))
  print(vlmc.to_json())

  print(vlmc.negative_log_likelihood("ABABBABA"))
  print([str(v) for v in VLMC.from_json_dir('../../trees')])
