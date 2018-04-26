#! /usr/bin/python3.6
import argparse
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import matplotlib as mpl

from vlmc import VLMC
from distance import NegativeLogLikelihood, NaiveParameterSampling, StationaryDistribution,\
    ACGTContent, FrobeniusNorm, EstimateVLMC, FixedLengthSequenceKLDivergence, Projection, PSTMatching
import parse_trees_to_json
from get_signature_metadata import get_metadata_for
from util.print_distance import print_metrics, print_distance_output
from util.distance_metrics import update_metrics, normalise_metrics
from util.draw_distance import plot_distance, update_box_plot_data, draw_box_plot

label_size = 20
mpl.rcParams['xtick.labelsize'] = label_size
mpl.rcParams['ytick.labelsize'] = label_size
mpl.rcParams['axes.axisbelow'] = True
mpl.rcParams['font.size'] = 24


def test_kl_divergence(tree_dir, out_dir, fixed_length):
  d = FixedLengthSequenceKLDivergence(fixed_length)
  test_distance_function(d, tree_dir, out_dir)


def test_distance_function(d, tree_dir, out_dir):
  parse_trees_to_json.parse_trees(tree_dir)
  vlmcs = VLMC.from_json_dir(tree_dir)

  metadata = get_metadata_for([vlmc.name for vlmc in vlmcs])

  test_dir = tree_dir + "_test"
  if os.path.isdir(test_dir):
    parse_trees_to_json.parse_trees(test_dir)
    test_vlmcs = VLMC.from_json_dir(test_dir)
  else:
    test_vlmcs = vlmcs

  try:
    os.stat(out_dir)
  except:
    os.mkdir(out_dir)

  return test_distance_function_(d, vlmcs, test_vlmcs, metadata, out_dir, True, False, True)


def test_distance_function_(d, vlmcs, test_vlmcs, metadata, out_dir, print_metrics=True, print_every_distance=False, do_plot=False):
  metrics = {
      "average_procent_of_genus_in_top": 0.0,
      "average_procent_of_family_in_top": 0.0,
      "total_average_distance_to_genus": 0.0,
      "total_average_distance_to_family": 0.0,
      "total_average_distance": 0.0,
      "global_time": 0
  }

  gc_distance_function = ACGTContent(['C', 'G'])

  all_gc_differences = np.empty((len(vlmcs), len(vlmcs)))
  all_family_orders = np.empty((len(vlmcs), len(vlmcs)))
  all_genus_orders = np.empty((len(vlmcs), len(vlmcs)))

  for f, vlmc in enumerate(vlmcs):
    sorted_results, elapsed_time = calculate_distances(d, vlmc, vlmcs)

    metrics = update_metrics(vlmc, vlmcs, sorted_results, metadata, elapsed_time, metrics)
    update_box_plot_data(vlmc, i, sorted_results, all_gc_differences,
                         all_family_orders, all_genus_orders, gc_distance_function, metadata)

    if print_every_distance:
      print_distance_output(vlmc, vlmcs, sorted_results, elapsed_time, metadata, metrics)
    if do_plots:
      plot_distance(sorted_results, vlmc, gc_distance_function, metadata, out_dir, True, False)

  if do_plts:
    number_of_bins = len(vlmcs) / 10
    draw_box_plot(all_gc_differences, all_family_orders, all_genus_orders, number_of_bins, out_dir)

  metrics = normalise_metrics(metrics, vlmcs)

  if print_metrics:
    print_metrics(metrics)

  return metrics


def calculate_distances(d, vlmc, other_vlmcs):
  start_time = time.time()
  distances = list(map(lambda other: d.distance(vlmc, other), other_vlmcs))
  elapsed_time = time.time() - start_time

  sorted_results = sorted(zip(distances, other_vlmcs), key=lambda t: t[0])

  return sorted_results, elapsed_time


def parse_distance_method(args):
  if args.negative_log_likelihood:
    print('Testing negative log likelihood with a generated sequence of length {}'.format(args.seqlen))
    return NegativeLogLikelihood(args.seqlen)
  elif args.parameter_sampling:
    print('Testing the measure of estimation error distance function, the parameter based sampling.')
    return NaiveParameterSampling()
  elif args.acgt_content:
    print("Testing distance based only on acgt content.")
    return ACGTContent(['C', 'G'])
  elif args.stationary_distribution:
    print("Testing distance based on the stationary distribution")
    return StationaryDistribution()
  elif args.estimate_vlmc:
    print("Testing distance with an estimated vlmc")
    inner_d = FrobeniusNorm()
    return EstimateVLMC(inner_d)
  elif args.frobenius_norm:
    print("Testing distance as frobenius norm")
    return FrobeniusNorm()
  elif args.kmeans:
    return Projection()
  elif args.pst_matching:
    return PSTMatching(args.dissimilarity_weight)
  else:
    return FrobeniusNorm()


def test(args):
  d = parse_distance_method(args)
  try:
    os.stat(args.out_directory)
  except:
    os.mkdir(args.out_directory)

  test_distance_function(d, args.directory, args.out_directory)

if __name__ == '__main__':
  parser = argparse.ArgumentParser(
      description='Tests the distance functions for the vlmcs in ../trees, checking which vlmc they most closely match.')

  parser.add_argument('--parameter-sampling', action='store_true')
  parser.add_argument('--negative-log-likelihood', action='store_true')
  parser.add_argument('--acgt-content', action='store_true')
  parser.add_argument('--stationary-distribution', action='store_true')
  parser.add_argument('--frobenius-norm', action='store_true')
  parser.add_argument('--estimate-vlmc', action='store_true')
  parser.add_argument('--fixed-length-kl-divergence', action='store_true')
  parser.add_argument('--kmeans', action='store_true')
  parser.add_argument('--pst-matching', action='store_true')

  parser.add_argument('--fixed-sequence-length', type=int, default=8,
                      help='The length of the strings that are used in the fixed sequence length KL-divergence method.')
  parser.add_argument('--seqlen', type=int, default=1000,
                      help='The length of the sequences that are generated to calculate the likelihood.')

  parser.add_argument('--directory', type=str, default='../trees',
                      help='The directory which contains the trees to be used.')
  parser.add_argument('--out-directory', type=str, default='../images',
                      help='The directory to where images are written.')
  parser.add_argument('--dissimilarity_weight', type=float, default=0.5)

  args = parser.parse_args()
  test(args)
