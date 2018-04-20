import os

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.colors as colors
import matplotlib.cm as cmx
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import random


def draw_graph(clustering_metrics, clusters, out_directory):
  G = clustering_metrics.G
  metadata = clustering_metrics.metadata

  cluster_colors = _plot_graph(G, metadata, clusters, out_directory)

  _plot_silhouette(clustering_metrics, cluster_colors, out_directory)


def _plot_graph(G, metadata, clusters, out_directory):
  families = sorted(list(set([m['family'] for m in metadata.values()])))
  genera = sorted(list(set([m['genus'] for m in metadata.values()])))
  genera_colors = [genera.index(metadata[v.name]['genus']) for v in G.nodes()]
  family_colors = [families.index(metadata[v.name]['family']) for v in G.nodes()]
  genera_colormap = plt.cm.Set1
  family_colormap = plt.cm.gist_stern

  pos = _draw_nodes(G, metadata, genera, families, genera_colors,
                    family_colors, genera_colormap, family_colormap)

  cluster_colors = _draw_clusters(G, pos)

  _draw_legend(G, metadata, genera, genera_colors, genera_colormap,
               families, family_colors, family_colormap)

  out_file = os.path.join(out_directory, 'clustering_{}.pdf'.format(clusters))
  plt.savefig(out_file, dpi='figure', format='pdf')
  plt.close()

  return cluster_colors


def _draw_nodes(G, metadata, genera, families, genera_colors, family_colors, genera_colormap, family_colormap):
  labels = {v: metadata[v.name]['species'] for v in G.nodes()}

  plt.figure(figsize=(30, 20), dpi=80)
  pos = graphviz_layout(G, prog='sfdp', args='-x -Goverlap=scale')

  nx.draw(G, pos, with_labels=False, labels=labels, width=1,
          font_size=16, node_color='w', edge_color='#cccccc')
  # families
  outer_circles = nx.draw_networkx_nodes(
      G, pos, node_size=2000, node_color=family_colors, cmap=family_colormap)
  outer_circles.set_edgecolor('black')

  # genera
  inner_circles = nx.draw_networkx_nodes(
      G, pos, node_size=600, node_color=genera_colors, cmap=genera_colormap)

  # slightly move the labels above of the nodes
  pos_higher = {}
  for k, v in pos.items():
    pos_higher[k] = (v[0], v[1] + 30)

  nx.draw_networkx_labels(G, pos_higher, labels)

  return pos


def _draw_clusters(G, pos):
  cluster_colormap = plt.cm.nipy_spectral

  connected_components_subgraphs = list(nx.connected_component_subgraphs(G))
  # Each clusters gets a random color, such that each cluster will be
  # easier to distinguish from its neighbours
  color_indecis = list(range(len(connected_components_subgraphs)))
  random.shuffle(color_indecis)
  cluster_colors = [cluster_colormap(i / len(connected_components_subgraphs))
                    for i in color_indecis]

  for i, subgraph in enumerate(connected_components_subgraphs):
    edge_width = 4.0  # to make the cluster colors easier to see
    nx.draw_networkx_edges(subgraph, pos, width=edge_width, edge_color=[
                           cluster_colors[i]] * len(subgraph.edges))

  return cluster_colors


def _draw_legend(G, metadata, genera, genera_colors, genera_colormap, families, family_colors, family_colormap):
  family_genus_combinations = sorted(
      set([(metadata[v.name]['family'], metadata[v.name]['genus']) for v in G.nodes()]))

  genera_norm = colors.Normalize(vmin=min(genera_colors), vmax=max(genera_colors))
  genera_colormap_mappable = cmx.ScalarMappable(norm=genera_norm, cmap=genera_colormap)

  family_norm = colors.Normalize(vmin=min(family_colors), vmax=max(family_colors))
  family_colormap_mappable = cmx.ScalarMappable(norm=family_norm, cmap=family_colormap)

  legend_markers = [Line2D([0],
                           [0],
                           label="Family: {} Genus: {}".format(family, genus),
                           marker='o',
                           markersize=20,
                           markeredgewidth=6,
                           markerfacecolor=genera_colormap_mappable.to_rgba(genera.index(genus)),
                           markeredgecolor=family_colormap_mappable.to_rgba(families.index(family))
                           ) for family, genus in family_genus_combinations]

  l = plt.legend(handles=legend_markers, fontsize=20)
  l.draggable()


def _plot_silhouette(clustering_metrics, cluster_colors, out_directory):
  G = clustering_metrics.G
  metadata = clustering_metrics.metadata
  silhouette = clustering_metrics.silhouette_metric()
  bar_heights = []
  bar_labels = []
  bar_colors = []
  connected_components = list(nx.connected_components(G))

  for i, cluster in enumerate(connected_components):
    for species in cluster:
      bar_heights.append(silhouette[species.name])
      bar_labels.append(metadata[species.name]['species'])
      bar_colors.append(cluster_colors[i])

  plt.figure(figsize=(30, 20), dpi=80)
  plt.bar(x=range(len(silhouette)), height=bar_heights, tick_label=bar_labels, color=bar_colors)
  plt.xticks(range(len(bar_heights)), rotation=30, ha="right")

  out_file = os.path.join(out_directory, 'sillhouette_{}.pdf'.format(len(connected_components)))
  plt.savefig(out_file, dpi='figure', format='pdf')
  plt.close()
