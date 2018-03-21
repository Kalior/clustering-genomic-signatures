import os

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.colors as colors
import matplotlib.cm as cmx
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout


def draw_graph(G, metadata):
  families = sorted(list(set([m['family'] for m in metadata.values()])))
  genera = sorted(list(set([m['genus'] for m in metadata.values()])))

  genera_colors = [genera.index(metadata[v.name]['genus']) for v in G.nodes()]
  family_colors = [families.index(metadata[v.name]['family']) for v in G.nodes()]
  labels = {v: metadata[v.name]['species'] for v in G.nodes()}

  genera_colormap = plt.cm.Set1
  family_colormap = plt.cm.gist_stern

  plt.figure(figsize=(30, 20), dpi=80)
  pos = graphviz_layout(G, prog='sfdp')
  nx.draw(G, pos, with_labels=False, labels=labels, width=1,
          font_size=16, node_color='w', edge_color='#ff7f00')
  nx.draw_networkx_nodes(G, pos, node_size=2000, node_color=family_colors, cmap=family_colormap)
  nx.draw_networkx_nodes(G, pos, node_size=600, node_color=genera_colors, cmap=genera_colormap)

  pos_higher = {}
  for k, v in pos.items():
    pos_higher[k] = (v[0], v[1] + 60)

  nx.draw_networkx_labels(G, pos_higher, labels)

  family_genus_combinations = sorted(
      set([(metadata[v.name]['family'], metadata[v.name]['genus']) for v in G.nodes()]))

  genera_norm = colors.Normalize(vmin=min(genera_colors), vmax=max(genera_colors))
  genera_colormap_mappable = cmx.ScalarMappable(norm=genera_norm, cmap=genera_colormap)

  family_norm = colors.Normalize(vmin=min(family_colors), vmax=max(family_colors))
  family_colormap_mappable = cmx.ScalarMappable(norm=family_norm, cmap=family_colormap)

  legend_markers = [Line2D([0],
                           [0],
                           label="Family: {:20}{:1}".format(family, genus),
                           marker='o',
                           markersize=20,
                           markeredgewidth=6,
                           markerfacecolor=genera_colormap_mappable.to_rgba(genera.index(genus)),
                           markeredgecolor=family_colormap_mappable.to_rgba(
      families.index(family))
  ) for family, genus in family_genus_combinations]

  l = plt.legend(handles=legend_markers, fontsize=20)
  l.draggable()

  out_file = os.path.join('../images', 'clustering.pdf')
  plt.savefig(out_file, dpi='figure', format='pdf')
