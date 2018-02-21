#! /usr/bin/python3.6
from vlmc import VLMC
from distance import NegativeLogLikelihood, NaiveParameterSampling, StationaryDistribution, ACGTContent
import parse_trees_to_json
import argparse
import time
from get_signature_metadata import get_metadata_for


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


def test_distance_function(d, tree_dir):
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

  extra_distance = ACGTContent(['C', 'G'])
  result_list = [output_line(metadata, vlmc, dist, v, extra_distance)
                 for (dist, v) in sorted_results]

  print('\n'.join(result_list) + '\n\n')


def output_line(metadata, vlmc, dist, v, d):
  return "{:>55}  {:20} {:20} GC-distance: {:7.5f}   distance: {:10.5f}  {}".format(
      metadata[v.name]['species'],
      metadata[v.name]['genus'],
      metadata[v.name]['family'],
      d.distance(vlmc, v),
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
    print("Testing distance based on the stationary distribution")
    test_stationary_distribution(args.directory)
