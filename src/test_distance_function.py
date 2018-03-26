#! /usr/bin/python3.6
import argparse
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import matplotlib as mpl

from vlmc import VLMC
from distance import NegativeLogLikelihood, NaiveParameterSampling, StationaryDistribution, ACGTContent, FrobeniusNorm, EstimateVLMC
import parse_trees_to_json
from get_signature_metadata import get_metadata_for
from util.print_distance import print_metrics, print_distance_output
from util.distance_metrics import update_metrics, normalise_metrics
from util.draw_distance import draw_gc_plot, plot_distance, update_box_plot_data, draw_box_plot

label_size = 20
mpl.rcParams['xtick.labelsize'] = label_size
mpl.rcParams['ytick.labelsize'] = label_size
mpl.rcParams['axes.axisbelow'] = True
mpl.rcParams['font.size'] = 24


def test_negloglike(tree_dir, sequence_length):
  d = NegativeLogLikelihood(sequence_length)
  test_distance_function(d, tree_dir)


def test_parameter_sampling(tree_dir):
  d = NaiveParameterSampling()
  test_distance_function(d, tree_dir)


def test_acgt_content(tree_dir):
  d = ACGTContent()
  test_distance_function(d, tree_dir)


def test_stationary_distribution(tree_dir):
  d = StationaryDistribution()
  test_distance_function(d, tree_dir)


def test_frobenius_norm(tree_dir):
  d = FrobeniusNorm()
  test_distance_function(d, tree_dir)


def test_estimate_vlmc(tree_dir, sequence_length):
  inner_d = FrobeniusNorm()
  d = EstimateVLMC(inner_d)
  test_distance_function(d, tree_dir)


def test_distance_function(d, tree_dir):
  parse_trees_to_json.parse_trees(tree_dir)
  vlmcs = VLMC.from_json_dir(tree_dir)
  metadata = get_metadata_for([vlmc.name for vlmc in vlmcs])

  test_dir = tree_dir + "_test"
  if os.path.isdir(test_dir):
    parse_trees_to_json.parse_trees(test_dir)
    test_vlmcs = VLMC.from_json_dir(test_dir)
  else:
    test_vlmcs = vlmcs

  image_dir = '../images'

  metrics = {
      "average_procent_of_genus_in_top": 0.0,
      "average_procent_of_family_in_top": 0.0,
      "total_average_distance_to_genus": 0.0,
      "total_average_distance_to_family": 0.0,
      "total_average_distance": 0.0,
      "global_time": 0
  }

  gc_distance_function = ACGTContent(['C', 'G'])
  fig, [distance_ax, gc_ax] = plt.subplots(2, sharex='col', figsize=(30, 20), dpi=80)
  distance_ax.set_xlim(-1, len(vlmcs))
  gc_ax.set_xlim(-1, len(vlmcs))
  plt.xticks(range(len(vlmcs)))
  gc_ax.grid(color='#cccccc', linestyle='--', linewidth=1)
  distance_ax.grid(color='#cccccc', linestyle='--', linewidth=1)

  all_gc_differences = np.empty((len(vlmcs), len(vlmcs)))
  all_family_orders = np.empty((len(vlmcs), len(vlmcs)))
  all_genus_orders = np.empty((len(vlmcs), len(vlmcs)))

  for i, vlmc in enumerate(vlmcs):
    start_time = time.time()
    distances = list(map(lambda other: d.distance(vlmc, other), vlmcs))
    elapsed_time = time.time() - start_time

    sorted_results = sorted(zip(distances, vlmcs),
                            key=lambda t: (t[0], metadata[t[1].name]['genus']))

    metrics = update_metrics(vlmc, vlmcs, sorted_results, metadata, elapsed_time, metrics)
    update_box_plot_data(vlmc, i, sorted_results, all_gc_differences,
                         all_family_orders, all_genus_orders, gc_distance_function, metadata)

    print_distance_output(vlmc, vlmcs, sorted_results, elapsed_time, metadata, metrics)
    draw_gc_plot(sorted_results, vlmc, gc_distance_function, distance_ax, gc_ax)
    plot_distance(sorted_results, vlmc, gc_distance_function, metadata, image_dir)

  number_of_bins = len(vlmcs) / 10
  draw_box_plot(all_gc_differences, all_family_orders, all_genus_orders, number_of_bins, image_dir)
  metrics = normalise_metrics(metrics, vlmcs)
  print_metrics(metrics)

  out_file = os.path.join(image_dir, 'distance.pdf')
  fig.savefig(out_file, dpi='figure', format='pdf')


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
      description='Tests the distance functions for the vlmcs in ../trees, checking which vlmc they most closely match.')

  parser.add_argument('--parameter-sampling', action='store_true')
  parser.add_argument('--negative-log-likelihood', action='store_true')
  parser.add_argument('--acgt-content', action='store_true')
  parser.add_argument('--stationary-distribution', action='store_true')
  parser.add_argument('--frobenius-norm', action='store_true')
  parser.add_argument('--estimate-vlmc', action='store_true')

  parser.add_argument('--seqlen', type=int, default=1000,
                      help='The length of the sequences that are generated to calculate the likelihood.')

  parser.add_argument('--directory', type=str, default='../trees',
                      help='The directory which contains the trees to be used.')

  args = parser.parse_args()

  if (args.negative_log_likelihood):
    print('Testing negative log likelihood with a generated sequence of length {}'.format(args.seqlen))
    test_negloglike(args.directory, args.seqlen)

  if (args.parameter_sampling):
    print('Testing the measure of estimation error distance function, the parameter based sampling.')
    test_parameter_sampling(args.directory)

  if (args.acgt_content):
    print("Testing distance based only on acgt content.")
    test_acgt_content(args.directory)

  if (args.stationary_distribution):
    print("Testing distance based on the stationary distribution.")
    test_stationary_distribution(args.directory)

  if (args.frobenius_norm):
    print("Testing distance frobenius norm.")
    test_frobenius_norm(args.directory)

  if (args.estimate_vlmc):
    print("Testing distance with an estimated vlmc")
    test_estimate_vlmc(args.directory, args.seqlen)
