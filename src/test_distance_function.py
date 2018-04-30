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
from util.draw_distance import plot_distance, update_gc_box_data, update_metadata_box,\
    plot_cummlative_box, plot_gc_box

label_size = 20 * 2
mpl.rcParams['xtick.labelsize'] = label_size
mpl.rcParams['ytick.labelsize'] = label_size
mpl.rcParams['axes.axisbelow'] = True
mpl.rcParams['font.size'] = 24 * 2


def test_distance_function(d, tree_dir, out_dir, plot_distances=False, plot_boxes=False):
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

  return test_distance_function_(d, vlmcs, test_vlmcs, metadata, out_dir, True, False, plot_distances, plot_boxes)


def test_distance_function_(d, vlmcs, test_vlmcs, metadata, out_dir, do_print_metrics=True, print_every_distance=False, plot_distances=False, plot_boxes=False):
  metrics = {
      "distance_name": d.__class__.__name__,
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

  for index, vlmc in enumerate(vlmcs):
    sorted_results, elapsed_time = calculate_distances(d, vlmc, vlmcs)

    metrics = update_metrics(vlmc, vlmcs, sorted_results, metadata, elapsed_time, metrics)
    update_gc_box_data(vlmc, index, sorted_results, all_gc_differences, gc_distance_function)
    update_metadata_box(vlmc, index, sorted_results, all_family_orders, metadata, 'family')
    update_metadata_box(vlmc, index, sorted_results, all_genus_orders, metadata, 'genus')

    if print_every_distance:
      print_distance_output(vlmc, vlmcs, sorted_results, elapsed_time, metadata, metrics)
    if plot_distances:
      plot_distance(sorted_results, vlmc, gc_distance_function, metadata, out_dir, True, False)

  if plot_boxes:
    number_of_bins = 10  # len(vlmcs) / 10
    plot_cummlative_box(all_family_orders, number_of_bins, 'family', out_dir)
    plot_cummlative_box(all_genus_orders, number_of_bins, 'genus', out_dir)
    plot_gc_box(all_gc_differences, number_of_bins, out_dir)

  metrics = normalise_metrics(metrics, vlmcs)

  if do_print_metrics:
    print_metrics(metrics, True)

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
    print("Testing distance as frobenius norm, with union: {}".format(args.use_union))
    return FrobeniusNorm(args.use_union)
  elif args.kmeans:
    return Projection()
  elif args.pst_matching:
    print("Testing distance with PST matching")
    return PSTMatching(args.dissimilarity_weight)
  elif args.fixed_length_kl_divergence:
    print("Testing distance with fixed length kl, with {} length".format(args.fixed_sequence_length))
    return FixedLengthSequenceKLDivergence(args.fixed_sequence_length)
  else:
    return FrobeniusNorm(args.use_union)


def test(args):
  d = parse_distance_method(args)
  try:
    os.stat(args.out_directory)
  except:
    os.mkdir(args.out_directory)

  test_distance_function(d, args.directory, args.out_directory,
                         args.plot_distances, args.plot_boxes)

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
  parser.add_argument('--dissimilarity_weight', type=float, default=0.5)
  parser.add_argument('--use-union', action='store_true')

  parser.add_argument('--directory', type=str, default='../trees',
                      help='The directory which contains the trees to be used.')
  parser.add_argument('--out-directory', type=str, default='../images',
                      help='The directory to where images are written.')
  parser.add_argument('--plot-distances', action='store_true')
  parser.add_argument('--plot-boxes', action='store_true')

  args = parser.parse_args()
  test(args)
