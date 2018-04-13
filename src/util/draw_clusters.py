import os

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.colors as colors
import matplotlib.cm as cmx
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import random


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
                           label="Family: {:20} Genus: {}".format(family, genus),
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


def draw_silhouette(clustering_metrics):
  # TODO would like to show which bars belong to which cluster in the other graph
  G = clustering_metrics.G
  metadata = clustering_metrics.metadata
  silhouette = clustering_metrics.silhouette_metric()
  bar_heights = []
  bar_labels = []
  cluster_colors = []
  cm = plt.get_cmap('gist_rainbow')
  connected_components = list(nx.connected_components(G))
  color_indecis = list(range(len(connected_components)))
  # Each clusters gets a random color.  If they are not random,
  # clusters that are next to each other are hard to distinguish
  random.shuffle(color_indecis)

  for i, cluster in enumerate(connected_components):
    for species in cluster:
      bar_heights.append(silhouette[species.name])
      bar_labels.append(metadata[species.name]['species'])
      cluster_colors.append(cm(color_indecis[i] / len(connected_components)))
      
  plt.figure(figsize=(30, 20), dpi=80)
  plt.bar(x=range(len(silhouette)), height=bar_heights, tick_label=bar_labels, color=cluster_colors)
  plt.xticks(range(len(bar_heights)), rotation=30, ha="right")
  plt.show()
