#! /usr/bin/python3.6
from vlmc import VLMC
from distance import NegativeLogLikelihood, NaiveParameterSampling, StationaryDistribution, ACGTContent
from clustering import GraphBasedClustering
import parse_trees_to_json
import argparse
import time
from get_signature_metadata import get_metadata_for


def test_negloglike(sequence_length):
  d = NegativeLogLikelihood(sequence_length)
  test_clustering(d, 0.02)


def test_parameter_sampling():
  d = NaiveParameterSampling()
  test_clustering(d, 0.2)


def test_acgt_content():
  d = ACGTContent()
  test_clustering(d, 0.2)


def test_stationary_distribution():
  d = StationaryDistribution()
  test_clustering(d, 0.2)


def test_clustering(d, threshold):
  tree_dir = "../trees"
  parse_trees_to_json.parse_trees(tree_dir)
  vlmcs = VLMC.from_json_dir(tree_dir)
  clustering = GraphBasedClustering(threshold, vlmcs, d)
  clustering.cluster()

if __name__ == '__main__':
  parser = argparse.ArgumentParser(
      description='Tests the distance functions for the vlmcs in ../trees, checking which vlmc they most closely match.')

  parser.add_argument('--parameter-sampling', action='store_true')
  parser.add_argument('--negative-log-likelihood', action='store_true')
  parser.add_argument('--acgt-content', action='store_true')
  parser.add_argument('--stationary-distribution', action='store_true')

  parser.add_argument('--seqlen', type=int, default=1000,
                      help='The length of the sequences that are generated to calculate the likelihood.')

  args = parser.parse_args()

  if (args.negative_log_likelihood):
    print('Testing negative log likelihood with a generated sequence of length {}'.format(args.seqlen))
    test_negloglike(args.seqlen)

  if (args.parameter_sampling):
    print('Testing the measure of estimation error distance function, the parameter based sampling.')
    test_parameter_sampling()

  if (args.acgt_content):
    print("Testing distance based only on acgt content.")
    test_acgt_content()

  if (args.stationary_distribution):
    print("Testing distance based on the stationary distribution")
    test_stationary_distribution()
