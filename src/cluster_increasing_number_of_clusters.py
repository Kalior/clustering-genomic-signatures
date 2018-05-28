#! /usr/bin/python3.6
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
import numpy as np
import os
import argparse

label_size = 20 * 3
mpl.rcParams['xtick.labelsize'] = label_size
mpl.rcParams['ytick.labelsize'] = label_size
mpl.rcParams['axes.axisbelow'] = True
mpl.rcParams['font.size'] = 24 * 3

from vlmc import VLMC
from distance import FrobeniusNorm, PSTMatching, NegativeLogLikelihood, ACGTContent
from clustering import AverageLinkClustering, MSTClustering
import parse_trees_to_json
from get_signature_metadata import get_metadata_for
from test_clustering import parse_clustering_method, add_clustering_arguments
from test_distance_function import parse_distance_method, add_distance_arguments


def test_clustering(d, vlmcs, cluster_class):
  metadata = get_metadata_for([vlmc.name for vlmc in vlmcs])

  metrics = np.zeros([len(vlmcs), 7], dtype=np.float32)

  clustering = cluster_class(vlmcs, d, metadata)
  for i in range(len(vlmcs) - 1, 0, -1):
    print(i)
    clustering_metrics = clustering.cluster(i)

    metrics[i, 0] = clustering_metrics.average_silhouette()
    metrics[i, 1] = clustering_metrics.average_percent_same_taxonomy('organism')
    metrics[i, 2] = clustering_metrics.average_percent_same_taxonomy('family')
    metrics[i, 3] = clustering_metrics.average_percent_same_taxonomy('genus')
    fam_sensitivity, fam_specificity = clustering_metrics.sensitivity_specificity('family')
    metrics[i, 4] = fam_sensitivity
    metrics[i, 5] = fam_specificity
    metrics[i, 6] = clustering_metrics.get_latest_merge_distance()

  return metrics


def plot_metrics(metrics, out_directory, name):
  fig, ax = plt.subplots(1, sharex='col', figsize=(30, 20), dpi=80)
  ax.set_title(
      '{}'.format(name))

  xticks_step = (len(metrics) // 20) + 1
  xtick_locs = np.arange(0, len(metrics), xticks_step)
  xtick_labels = np.arange(1, len(metrics), xticks_step)

  ax.set_xticks(xtick_locs)
  ax.set_xticklabels(xtick_labels)

  ax.set_xlim(-1, len(metrics))
  ax.set_ylim(0, 1.1)
  ax.set_xlabel('Number of clusters')
  ax.set_ylabel('Average of metric')
  ax.grid(color='#cccccc', linestyle='--', linewidth=1)

  linestyles = ['-', '--', '-.', ':', '--', '-.', ':']
  linewidths = [4, 4, 4, 4, 7, 7, 7]
  colors = ['#e53935', '#8E24AA', '#3949AB', '#039BE5',
            '#00897B', '#7CB342', '#546E7A', '#FB8C00', '#6D4C41']

  for i in range(metrics.shape[1]):
    ax.plot(metrics[1:, i], markersize=0, marker=None, color=colors[i],
            linestyle=linestyles[i], linewidth=linewidths[i])

  labels = ['Silhouette',
            'Percent of organism',
            'Percent of family',
            'Percent of genus',
            'Sensitivity of family',
            'Specificity of family',
            'Distance between merged clusters'
            ]

  legend_markers = [legend_marker(labels[i], linestyles[i], colors[i], linewidths[i])
                    for i in range(metrics.shape[1])]

  ax.legend(handles=legend_markers, fontsize=30, markerscale=0, loc=4)

  out_file = os.path.join(
      out_directory, '{}.pdf'.format(name))
  plt.savefig(out_file, dpi='figure', format='pdf')
  plt.close(fig)


def legend_marker(label, linestyle, color, linewidth):
  return Line2D([0], [0], marker=None, linestyle=linestyle, linewidth=linewidth,
                markerfacecolor=color, color=color, label=label)


def test(args):
  tree_directory = args.directory
  out_directory = args.out_directory
  parse_trees_to_json.parse_trees(tree_directory)
  vlmcs = VLMC.from_json_dir(tree_directory)

  cluster_class = parse_clustering_method(args)
  d = parse_distance_method(args)

  if args.name:
    name = args.name
  else:
    name = cluster_class.__name__ + ", " + d.__class__.__name__

  metrics = test_clustering(d, vlmcs, cluster_class)

  try:
    os.stat(out_directory)
  except:
    os.mkdir(out_directory)

  plot_metrics(metrics, out_directory, name)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
      description=('Outputs a plot the clustering metrics with number of clusters from 1 to size of data set.'))

  add_distance_arguments(parser)
  add_clustering_arguments(parser)

  parser.add_argument('--directory', type=str, default='../trees_more_192',
                      help='The directory to source the trees for the VLMCs from.')
  parser.add_argument('--out-directory', type=str, default='../images/tests/increasing-clusters/',
                      help='The directory to where images are written.')

  parser.add_argument('--name', type=str)

  args = parser.parse_args()
  test(args)
