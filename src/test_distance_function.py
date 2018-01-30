#! /usr/bin/python
from vlmc import VLMC
from distance import NegativeLogLikelihood, NaiveParameterSampling
import parse_trees_to_json
import argparse


def test_negloglike(sequence_length):
  d = NegativeLogLikelihood(sequence_length)
  test_distance_function(d)


def test_parameter_sampling():
  d = NaiveParameterSampling()
  test_distance_function(d)


def test_distance_function(d):
  tree_dir = "../trees"
  parse_trees_to_json.parse_trees(tree_dir)
  vlmcs = VLMC.from_json_dir(tree_dir)
  for vlmc in vlmcs:
    distances = list(map(lambda other: d.distance(vlmc, other), vlmcs))
    closest_vlmc_i = distances.index(min(distances))
    closest_vlmc = vlmcs[closest_vlmc_i]
    print(vlmc.name + "\nis closest to\n" + closest_vlmc.name +
          "\n" + str(vlmc == closest_vlmc) + "\n\n")

if __name__ == '__main__':
  parser = argparse.ArgumentParser(
      description='Tests the distance functions for the vlmcs in ../trees, checking which vlmc they most closely match.')

  parser.add_argument('--parameter-sampling', action='store_true')
  parser.add_argument('--negative-log-likelihood', action='store_true')
  parser.add_argument('--seqlen', type=int, default=1000,
                      help='The length of the sequences that are generated to calculate the likelihood.')

  args = parser.parse_args()

  if (args.negative_log_likelihood):
    print('Testing negative log likelihood with a generated sequence of length {}'.format(args.seqlen))
    test_negloglike(args.seqlen)

  if (args.parameter_sampling):
    print('Testing the measure of estimation error distance function, the parameter based sampling.')
    test_parameter_sampling()
