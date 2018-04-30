import os

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.colors as colors
import matplotlib.cm as cmx
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import random


def draw_graph(clustering_metrics, meta_name, meta_key, clusters, out_directory):
  G = clustering_metrics.G
  metadata = clustering_metrics.metadata

  cluster_colors = _plot_graph(G, metadata, meta_name, meta_key, clusters, out_directory)

  _plot_silhouette(clustering_metrics, cluster_colors, meta_key, out_directory)


def _plot_graph(G, metadata, meta_name, meta_key, clusters, out_directory):
  meta = sorted(list(set([m[meta_key] for m in metadata.values()])))
  meta_colors = [meta.index(metadata[v.name][meta_key]) for v in G.nodes()]
  meta_colormap = plt.cm.nipy_spectral

  pos = _draw_nodes(G, metadata, meta, meta_colors, meta_colormap)

  cluster_colors = _draw_clusters(G, pos)

  _draw_legend(G, metadata, meta_name, meta, meta_colors, meta_colormap)

  out_file = os.path.join(out_directory, 'clustering_{}_{}.pdf'.format(meta_key, clusters))
  plt.savefig(out_file, dpi='figure', format='pdf')
  plt.close()

  return cluster_colors


def _draw_nodes(G, metadata, meta, meta_colors, meta_colormap):
  labels = {v: metadata[v.name]['species'] for v in G.nodes()}

  plt.figure(figsize=(30, 20), dpi=80)
  pos = graphviz_layout(G, prog='sfdp', args='-x -Goverlap=scale')

  nx.draw(G, pos, with_labels=False, labels=labels, width=1,
          font_size=16, node_color='w', edge_color='#cccccc')

  # meta
  inner_circles = nx.draw_networkx_nodes(
      G, pos, node_size=600, node_color=meta_colors, cmap=meta_colormap)
  inner_circles.set_edgecolor('black')

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


def _draw_legend(G, metadata, meta_name, meta, meta_colors, meta_colormap):
  meta_norm = colors.Normalize(vmin=min(meta_colors), vmax=max(meta_colors))
  meta_colormap_mappable = cmx.ScalarMappable(norm=meta_norm, cmap=meta_colormap)

  legend_markers = [Line2D([0],
                           [0],
                           label="{}: {}".format(meta_name, m),
                           marker='o',
                           markersize=20,
                           markeredgewidth=6,
                           markerfacecolor=meta_colormap_mappable.to_rgba(meta.index(m)),
                           markeredgecolor='#ffffff'
                           ) for m in meta]

  l = plt.legend(handles=legend_markers, fontsize=20)
  l.draggable()


def _plot_silhouette(clustering_metrics, cluster_colors, meta_key, out_directory):
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

  out_file = os.path.join(out_directory, 'sillhouette_{}_{}.pdf'.format(
      meta_key, len(connected_components)))
  plt.savefig(out_file, dpi='figure', format='pdf')
  plt.close()
