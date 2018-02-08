#!/usr/bin/python3.6
from vlmc import VLMC
from distance import NegativeLogLikelihood, NaiveParameterSampling, StationaryDistribution, ACGTContent, StatisticalMetric
import parse_trees_to_json
import argparse
import time
from get_signature_metadata import get_metadata_for


def test_negloglike(sequence_length):
  d = NegativeLogLikelihood(sequence_length)
  test_distance_function(d)


def test_parameter_sampling():
  d = NaiveParameterSampling()
  test_distance_function(d)

def test_statistical_metric():
  d = StatisticalMetric(1200, 0.05)
  test_distance_function(d)

def test_acgt_content():
  d = ACGTContent()
  test_distance_function(d)


def test_stationary_distribution():
  d = StationaryDistribution()
  test_distance_function(d)


def test_distance_function(d):
  tree_dir = "../trees"
  parse_trees_to_json.parse_trees(tree_dir)
  vlmcs = VLMC.from_json_dir(tree_dir)
  metadata = get_metadata_for([vlmc.name for vlmc in vlmcs])

  for vlmc in vlmcs:
    start_time = time.time()
    distances = list(map(lambda other: d.distance(vlmc, other), vlmcs))
    elapsed_time = time.time() - start_time

    test_output(vlmc, vlmcs, distances, elapsed_time, metadata)


def test_output(vlmc, vlmcs, distances, elapsed_time, metadata):
  closest_vlmc_i = distances.index(min(distances))
  closest_vlmc = vlmcs[closest_vlmc_i]

  print("{} matches self: {}.\nDistance calculated in: {}s\n".format(
      metadata[vlmc.name]['species'], vlmc == closest_vlmc, elapsed_time))

  sorted_results = sorted(zip(distances, vlmcs),
                          key=lambda t: (t[0], metadata[t[1].name]['genus']))

  result_list = [output_line(metadata, vlmc, dist, v) for (dist, v) in sorted_results]

  print('\n'.join(result_list) + '\n\n')


def output_line(metadata, vlmc, dist, v):
  return "{:>55}  {:20} {:20}  distance: {:10.5f}  {}".format(
      metadata[v.name]['species'],
      metadata[v.name]['genus'],
      metadata[v.name]['family'],
      dist,
      same_genus_or_family_string(metadata, vlmc, v))


def same_genus_or_family_string(metadata, vlmc, other_vlmc):
  if metadata[other_vlmc.name]['genus'] == metadata[vlmc.name]['genus']:
    return 'same genus'
  elif metadata[other_vlmc.name]['family'] == metadata[vlmc.name]['family']:
    return 'same family'
  else:
    return ''

if __name__ == '__main__':
  parser = argparse.ArgumentParser(
      description='Tests the distance functions for the vlmcs in ../trees, checking which vlmc they most closely match.')

  parser.add_argument('--parameter-sampling', action='store_true')
  parser.add_argument('--negative-log-likelihood', action='store_true')
  parser.add_argument('--acgt-content', action='store_true')
  parser.add_argument('--stationary-distribution', action='store_true')

  parser.add_argument('--seqlen', type=int, default=1000,
                      help='The length of the sequences that are generated to calculate the likelihood.')
  parser.add_argument('--statistical-metric', action='store_true')
  args = parser.parse_args()

  if (args.negative_log_likelihood):
    print('Testing negative log likelihood with a generated sequence of length {}'.format(args.seqlen))
    test_negloglike(args.seqlen)

  if (args.parameter_sampling):
    print('Testing the measure of estimation error distance function, the parameter based sampling.')
    test_parameter_sampling()

  if (args.statistical_metric):
    print('Testing statistical metric thing')
    test_statistical_metric()

  if (args.acgt_content):
    print("Testing distance based only on acgt content.")
    test_acgt_content()

  if (args.stationary_distribution):
    print("Testing distance based on the stationary distribution")
    test_stationary_distribution()

