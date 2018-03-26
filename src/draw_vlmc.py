#! /usr/bin/python3.6

import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from enum import Enum
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


def save(tree_dir, out_dir, deltas=False):
  parse_trees(tree_dir, deltas)
  vlmcs = VLMC.from_json_dir(tree_dir)
  metadata = get_metadata_for([vlmc.name for vlmc in vlmcs])

  try:
    os.stat(out_dir)
  except:
    os.mkdir(out_dir)

  for vlmc in vlmcs:
    plt.figure(figsize=(150, 30), dpi=80)
    draw_with_probabilities(vlmc, metadata, deltas)
    out_file = os.path.join(out_dir, vlmc.name + '.pdf')
    plt.axis('off')
    plt.savefig(out_file, dpi='figure', format='pdf', bbox_inches='tight')
    plt.close()


def draw_with_probabilities(vlmc, metadata, deltas):
  G = nx.DiGraph()
  if vlmc.name in metadata:
    root_name = metadata[vlmc.name]['species']
  else:
    root_name = vlmc.name
  G.add_node(root_name, inner=True, delta=-1)
  add_children(vlmc, G, "", root_name, deltas)

  pos = graphviz_layout(G, prog='dot')
  nx.draw_networkx_nodes(G, pos, node_size=10, node_color='w')

  if deltas:
    nodes = G.nodes(data=True)
    inner_nodes = {n[0]: n[1]['delta'] for n in nodes if n[1]['inner']}
  else:
    nodes = G.nodes(data='inner')
    inner_nodes = {n[0]: n[0] for n in nodes if n[1]}

  nx.draw_networkx_labels(G, pos, font_size=16, labels=inner_nodes)

  edges = G.edges(data='inner')
  inner_edges = [e for e in edges if e[2]]
  outer_edges = [e for e in edges if not e[2]]
  nx.draw_networkx_edges(G, pos, edgelist=inner_edges, arrows=True, edge_color='#ff7f00',
                         width=2, style='solid')

  if not deltas:
    nx.draw_networkx_edges(G, pos, edgelist=outer_edges, arrows=False, edge_color='#007fff',
                           width=1, style='dashed')

  if deltas:
    edge_attributes = ['symbol']
  else:
    edge_attributes = ['symbol', 'prob']

  attributes = [nx.get_edge_attributes(G, attr) for attr in edge_attributes]
  keys = set([k for attrs in attributes for k in attrs.keys()])
  labels = {k: " ".join([attrs[k] for attrs in attributes if k in attrs]) for k in keys}
  nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_size=12)


def add_children(vlmc, G, context, root_name, deltas):
  for c in vlmc.alphabet:
    child = c + context
    parent_label = context
    if parent_label == "":
      parent_label = root_name
    if child in vlmc.tree:
      G.add_node(child, inner=True, delta="{:.2f}".format(vlmc.tree[context][c]))
      G.add_edge(parent_label, child, symbol=c,
                 prob="{:.2f}".format(vlmc.tree[context][c]), inner=True)
      add_children(vlmc, G, child, root_name, deltas)
    elif not deltas:
      G.add_node(child, inner=False, delta=-1.0)
      G.add_edge(parent_label, child, symbol=c, prob="{:.2f}".format(
          vlmc.tree[context][c]), inner=False)


def save_intersection(tree_dir, out_dir):
  parse_trees(tree_dir)
  vlmcs = VLMC.from_json_dir(tree_dir)
  metadata = get_metadata_for([vlmc.name for vlmc in vlmcs])

  try:
    os.stat(out_dir)
  except:
    os.mkdir(out_dir)

  for vlmc_left, vlmc_right in combinations(vlmcs, 2):
    plt.figure(figsize=(90, 30), dpi=20)
    draw_intersection(vlmc_left, vlmc_right, metadata)
    out_file = os.path.join(out_dir, vlmc_left.name + '_' + vlmc_right.name + '.pdf')
    plt.title(metadata[vlmc_left.name]['species'] +
              " and " + metadata[vlmc_right.name]['species'])
    plt.axis('off')
    # plt.show()
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
  # tree_dir = '../original_test_trees'
  # image_dir = '../vlmc_illustrations'
  # save_intersection(tree_dir, image_dir)
  tree_dir = '../trees_pst_better'
  image_dir = '../images'
  save(tree_dir, image_dir, False)
