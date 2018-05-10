#! /usr/bin/python3.6
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os

label_size = 20 * 2
mpl.rcParams['xtick.labelsize'] = label_size
mpl.rcParams['ytick.labelsize'] = label_size
mpl.rcParams['axes.axisbelow'] = True
mpl.rcParams['font.size'] = 24 * 2

from vlmc import VLMC
from distance import FrobeniusNorm, PSTMatching, NegativeLogLikelihood, ACGTContent
from clustering import AverageLinkClustering, MSTClustering
import parse_trees_to_json
from get_signature_metadata import get_metadata_for


def test_clustering(d, vlmcs, cluster_class):
  metadata = get_metadata_for([vlmc.name for vlmc in vlmcs])

  metrics = np.zeros([len(vlmcs), 8], dtype=np.float32)

  clustering = cluster_class(vlmcs, d, metadata)
  for i in range(len(vlmcs) - 1, 0, -1):
    print(i)
    clustering_metrics = clustering.cluster(i)

    metrics[i, 0] = clustering_metrics.average_silhouette()
    metrics[i, 1] = clustering_metrics.average_percent_same_taxonomy('order')
    metrics[i, 2] = clustering_metrics.average_percent_same_taxonomy('family')
    metrics[i, 3] = clustering_metrics.average_percent_same_taxonomy('subfamily')
    metrics[i, 4] = clustering_metrics.average_percent_same_taxonomy('genus')
    fam_sensitivity, fam_specificity = clustering_metrics.sensitivity_specificity('family')
    metrics[i, 5] = fam_sensitivity
    metrics[i, 6] = fam_specificity
    metrics[i, 7] = clustering_metrics.average_percent_same_taxonomy('baltimore')

  return metrics


def plot_metrics(metrics, out_directory, name):
  fig, ax = plt.subplots(1, sharex='col', figsize=(30, 20), dpi=80)
  ax.set_title(
      'Metrics with increasing number of clusters, {}'.format(name))

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
            'Percent of order',
            'Percent of family',
            'Percent of subfamily',
            'Percent of genus',
            'Sensitivity of family',
            'Specificity of family',
            'Percent of baltimore'
            ]

  ax.legend(handles=handles, labels=labels, fontsize=30, markerscale=3)

  out_file = os.path.join(
      out_directory, 'increasing-number-of-clusters-{}.pdf'.format(name))
  plt.savefig(out_file, dpi='figure', format='pdf')
  plt.close(fig)


def test(tree_directory, out_directory):
  parse_trees_to_json.parse_trees(tree_directory)
  vlmcs = VLMC.from_json_dir(tree_directory)

  cluster_class = AverageLinkClustering
  # d = NegativeLogLikelihood(16000)
  # d = FrobeniusNorm(True)
  # d = PSTMatching(0.5)
  d = ACGTContent(['C', 'G'])
  name = 'GC-distance'

  metrics = test_clustering(d, vlmcs, cluster_class)

  try:
    os.stat(out_directory)
  except:
    os.mkdir(out_directory)

  plot_metrics(metrics, out_directory, name)


if __name__ == '__main__':
  tree_directory = '../trees_more_192'
  out_directory = '../images/tests/increasing-clusters/single'
  test(tree_directory, out_directory)
