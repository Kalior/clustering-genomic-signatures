#! /usr/bin/python3.6

import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt
import os

from vlmc import VLMC
from parse_trees_to_json import parse_trees
from get_signature_metadata import get_metadata_for


def save(tree_dir, out_dir):
  parse_trees(tree_dir)
  vlmcs = VLMC.from_json_dir(tree_dir)
  metadata = get_metadata_for([vlmc.name for vlmc in vlmcs])

  try:
    os.stat(out_dir)
  except:
    os.mkdir(out_dir)

  for vlmc in vlmcs:
    plt.figure(figsize=(150, 30), dpi=80)
    vlmc.draw(metadata)
    out_file = os.path.join(out_dir, vlmc.name + '.pdf')
    plt.axis('off')
    plt.savefig(out_file, dpi='figure', format='pdf', bbox_inches='tight')
    plt.close()


def draw(vlmc, metadata):
  G = nx.DiGraph()
  if vlmc.name in metadata:
    root_name = metadata[vlmc.name]['species']
  else:
    root_name = vlmc.name
  G.add_node(root_name, inner=True)
  add_children(vlmc, G, "", root_name)

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

  edge_attributes = ['symbol', 'prob']
  attributes = [nx.get_edge_attributes(G, attr) for attr in edge_attributes]
  keys = set([k for attrs in attributes for k in attrs.keys()])
  labels = {k: " ".join([attrs[k] for attrs in attributes if k in attrs]) for k in keys}
  nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_size=12)


def _add_children(vlmc, G, context, root_name):
  for c in vlmc.alphabet:
    child = c + context
    parent_label = context
    if parent_label == "":
      parent_label = root_name
    if child in vlmc.tree:
      G.add_node(child, inner=True)
      G.add_edge(parent_label, child, symbol=c,
                 prob="{:.2f}".format(vlmc.tree[context][c]), inner=True)
      vlmc._add_children(G, child, root_name)
    else:
      G.add_node(child, inner=False)
      G.add_edge(parent_label, child, symbol=c, prob="{:.2f}".format(
          vlmc.tree[context][c]), inner=False)


if __name__ == '__main__':
  tree_dir = '../trees_pst_better'
  image_dir = '../images'
  save(tree_dir, image_dir)
