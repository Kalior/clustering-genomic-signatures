#! /usr/bin/python3.6
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os

label_size = 20
mpl.rcParams['xtick.labelsize'] = label_size
mpl.rcParams['ytick.labelsize'] = label_size
mpl.rcParams['axes.axisbelow'] = True
mpl.rcParams['font.size'] = 24

from vlmc import VLMC
from distance import FrobeniusNorm
from clustering import AverageLinkClustering
import parse_trees_to_json
from get_signature_metadata import get_metadata_for


def test_clustering(d, vlmcs, cluster_class):
  metrics = np.zeros([len(vlmcs), 6], dtype=np.float32)

  metadata = get_metadata_for([vlmc.name for vlmc in vlmcs])

  clustering = cluster_class(vlmcs, d, metadata)
  for i in range(len(vlmcs) - 1, 0, -1):
    print(i)
    clustering_metrics = clustering.cluster(i)
    clustering_metrics.metadata = metadata

    metrics[i, 0] = clustering_metrics.average_silhouette()
    metrics[i, 1] = clustering_metrics.average_percent_same_taxonomy('organism')
    metrics[i, 2] = clustering_metrics.average_percent_same_taxonomy('order')
    metrics[i, 3] = clustering_metrics.average_percent_same_taxonomy('family')
    metrics[i, 4] = clustering_metrics.average_percent_same_taxonomy('subfamily')
    metrics[i, 5] = clustering_metrics.average_percent_same_taxonomy('genus')

  return metrics


def plot_metrics(metrics, out_directory):
  fig, ax = plt.subplots(1, sharex='col', figsize=(30, 20), dpi=80)
  ax.set_title('Metrics with increasing number of clusters', fontsize=30)

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

  handles = ax.plot(metrics[1:], markersize=5, marker='o')

  labels = ['Silhouette',
            'Percent of organism',
            'Percent of order',
            'Percent of family',
            'Percent of subfamily',
            'Percent of genus'
            ]

  ax.legend(handles=handles, labels=labels, bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)

  out_file = os.path.join(out_directory, 'increasing-number-of-clusters.pdf')
  plt.savefig(out_file, dpi='figure', format='pdf')
  plt.close(fig)


def test(tree_directory, out_directory):
  parse_trees_to_json.parse_trees(tree_directory)
  vlmcs = VLMC.from_json_dir(tree_directory)

  cluster_class = AverageLinkClustering
  d = FrobeniusNorm()

  metrics = test_clustering(d, vlmcs, cluster_class)
  plot_metrics(metrics, out_directory)


if __name__ == '__main__':
  tree_directory = '../trees_more_192'
  out_directory = '../images'
  test(tree_directory, out_directory)
