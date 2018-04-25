#! /usr/bin/python3.6

import argparse
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from enum import Enum, auto
from itertools import combinations

from vlmc import VLMC
from parse_trees_to_json import parse_trees
from get_signature_metadata import get_metadata_for

mpl.rcParams['font.size'] = 24


class Part(Enum):
  LEFT = 1
  INTERSECTION = 2
  RIGHT = 3
  NONE = 4


class NodeLabelType(Enum):
  CONTEXT = auto()
  DELTA_VALUE = auto()
  STATIONARY_PROBABILITY = auto()

  @staticmethod
  def get_type(deltas, stationary):
    if deltas:
      return NodeLabelType.DELTA_VALUE
    elif stationary:
      return NodeLabelType.STATIONARY_PROBABILITY
    else:
      return NodeLabelType.CONTEXT

  @staticmethod
  def get_root_label(label):
    return {
        NodeLabelType.CONTEXT: "",
        NodeLabelType.DELTA_VALUE: "",
        NodeLabelType.STATIONARY_PROBABILITY: "1"
    }.get(label)

  @staticmethod
  def get_node_label(vlmc, node, label_type):
    return {
        NodeLabelType.CONTEXT: node,
        NodeLabelType.DELTA_VALUE: "-1", # currently not used, since deltas are its own attributes
        NodeLabelType.STATIONARY_PROBABILITY: "{:.5f}".format(vlmc.stationary_probability(node))
    }.get(label_type)


def save(vlmcs, metadata, out_dir, deltas=False, stationary_prob_labels=False):
  print(NodeLabelType)
  label_type = NodeLabelType.get_type(deltas, stationary_prob_labels)
  for vlmc in vlmcs[0:2]:
    plt.figure(figsize=(150, 30), dpi=80)
    draw_with_probabilities(vlmc, metadata, label_type)
    out_file = os.path.join(out_dir, vlmc.name + '.pdf')
    plt.axis('off')
    plt.savefig(out_file, dpi='figure', format='pdf', bbox_inches='tight')
    plt.close()


def draw_with_probabilities(vlmc, metadata, label_type):
  G = nx.DiGraph()
  if vlmc.name in metadata:
    root_name = metadata[vlmc.name]['species']
  else:
    root_name = vlmc.name
  root_label = NodeLabelType.get_root_label(label_type)
  G.add_node(root_name, label=root_label, inner=True, delta=-1)
  add_children(vlmc, G, "", root_name, label_type)

  pos = graphviz_layout(G, prog='dot')
  nx.draw_networkx_nodes(G, pos, node_size=10, node_color='w')

  if label_type is NodeLabelType.DELTA_VALUE:
    nodes = G.nodes(data=True)
    inner_nodes = {n[0]: n[1]['delta'] for n in nodes if n[1]['inner']}
  else:
    nodes = G.nodes(data=True)
    inner_nodes = {n[0]: n[1]['label'] for n in nodes if n[1]['inner']}

  nx.draw_networkx_labels(G, pos, font_size=16, labels=inner_nodes)

  edges = G.edges(data='inner')
  inner_edges = [e for e in edges if e[2]]
  outer_edges = [e for e in edges if not e[2]]
  nx.draw_networkx_edges(G, pos, edgelist=inner_edges, arrows=True, edge_color='#ff7f00',
                         width=2, style='solid')

  if label_type is not NodeLabelType.DELTA_VALUE:
    nx.draw_networkx_edges(G, pos, edgelist=outer_edges, arrows=False, edge_color='#007fff',
                           width=1, style='dashed')

  if label_type is NodeLabelType.DELTA_VALUE:
    edge_attributes = ['symbol']
  else:
    edge_attributes = ['symbol', 'prob']

  attributes = [nx.get_edge_attributes(G, attr) for attr in edge_attributes]
  keys = set([k for attrs in attributes for k in attrs.keys()])
  labels = {k: " ".join([attrs[k] for attrs in attributes if k in attrs]) for k in keys}
  nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_size=12)


def add_children(vlmc, G, context, root_name, label_type):
  for c in vlmc.alphabet:
    child = c + context
    parent_label = context
    if parent_label == "":
      parent_label = root_name

    if child in vlmc.tree:
      new_label = NodeLabelType.get_node_label(vlmc, child, label_type)
      G.add_node(child, label=new_label, inner=True, delta="{:.2f}".format(vlmc.tree[context][c]))
      G.add_edge(parent_label, child, symbol=c,
                 prob="{:.2f}".format(vlmc.tree[context][c]), inner=True)
      add_children(vlmc, G, child, root_name, label_type)
    elif label_type is not NodeLabelType.DELTA_VALUE:
      G.add_node(child, label="", inner=False, delta=-1.0)
      G.add_edge(parent_label, child, symbol=c, prob="{:.2f}".format(
          vlmc.tree[context][c]), inner=False)


def save_intersection(vlmcs, metadata, out_dir):
  for vlmc_left, vlmc_right in combinations(vlmcs, 2):
    plt.figure(figsize=(90, 30), dpi=20)
    draw_intersection(vlmc_left, vlmc_right, metadata)
    out_file = os.path.join(out_dir, vlmc_left.name + '_' + vlmc_right.name + '.pdf')
    plt.title(metadata[vlmc_left.name]['species'] + " and " + metadata[vlmc_right.name]['species'])
    # plt.title(vlmc_left.name + " and " + vlmc_right.name)
    plt.axis('off')
    plt.savefig(out_file, dpi='figure', format='pdf', bbox_inches='tight')
    plt.close()


def draw_intersection(vlmc_left, vlmc_right, metadata):
  G = nx.DiGraph()
  root_name = " "
  G.add_node(root_name, part=Part.INTERSECTION)
  add_children_intersection(vlmc_left, vlmc_right, G, "", root_name)
  pos = graphviz_layout(G, prog='dot')

  nodes = G.nodes(data='part')
  intersection_nodes = [n[0] for n in nodes if n[1] == Part.INTERSECTION]
  left_nodes = [n[0] for n in nodes if n[1] == Part.LEFT]
  right_nodes = [n[0] for n in nodes if n[1] == Part.RIGHT]

  edges = G.edges(data='part')
  intersection_edges = [e for e in edges if e[2] == Part.INTERSECTION]
  left_edges = [e for e in edges if e[2] == Part.LEFT]
  right_edges = [e for e in edges if e[2] == Part.RIGHT]

  nx.draw_networkx_nodes(G, pos, nodelist=intersection_nodes, node_size=100, node_color='#ff7f00')
  nx.draw_networkx_nodes(G, pos, nodelist=left_nodes, node_size=100, node_color='#007fff')
  nx.draw_networkx_nodes(G, pos, nodelist=right_nodes, node_size=100, node_color='#ff007f')
  # nx.draw_networkx_labels(G, pos, font_size=16)

  nx.draw_networkx_edges(G, pos, edgelist=intersection_edges, arrows=True,
                         edge_color='#ff7f00', width=6, style='solid')
  nx.draw_networkx_edges(G, pos, edgelist=left_edges, arrows=False,
                         edge_color='#007fff', width=2, style='dashed')
  nx.draw_networkx_edges(G, pos, edgelist=right_edges, arrows=False,
                         edge_color='#ff007f', width=2, style='dashed')


def add_children_intersection(vlmc_left, vlmc_right, G, context, root_name):
  for c in vlmc_left.alphabet:
    child = c + context
    parent_label = context
    if parent_label == "":
      parent_label = root_name
    if child in vlmc_left.tree and child in vlmc_right.tree:
      part = Part.INTERSECTION
    elif child in vlmc_left.tree:
      part = Part.LEFT
    elif child in vlmc_right.tree:
      part = Part.RIGHT
    else:
      part = Part.NONE

    if part != Part.NONE:
      G.add_node(child, part=part)
      G.add_edge(parent_label, child, symbol=c, part=part)
      add_children_intersection(vlmc_left, vlmc_right, G, child, root_name)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
      description='Prints vlmcs models, or the intersection of such.')
  parser.add_argument('--deltas', action='store_true')
  parser.add_argument('--intersection', action='store_true')
  parser.add_argument('--stationary-probability-labels', action='store_true')

  parser.add_argument('--directory', type=str, default='../trees_pst_better',
                      help='The directory which contains the vlmcs to be printed.')
  parser.add_argument('--out-directory', type=str, default='../images',
                      help='The directory to where the images should be written.')

  args = parser.parse_args()

  try:
    os.stat(args.out_directory)
  except:
    os.mkdir(args.out_directory)

  parse_trees(args.directory, args.deltas)
  vlmcs = VLMC.from_json_dir(args.directory)
  metadata = get_metadata_for([vlmc.name for vlmc in vlmcs])

  if args.intersection:
    save_intersection(vlmcs, metadata, args.out_directory)
  else:
    save(vlmcs, metadata, args.out_directory, args.deltas, args.stationary_probability_labels)
