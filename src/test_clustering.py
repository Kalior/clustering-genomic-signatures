#! /usr/bin/python3.6
from vlmc import VLMC
from distance import NegativeLogLikelihood, NaiveParameterSampling, StationaryDistribution, ACGTContent, EstimateVLMC
from clustering import GraphBasedClustering
import parse_trees_to_json
import argparse
import time
from get_signature_metadata import get_metadata_for


def test_negloglike(sequence_length, clusters):
  d = NegativeLogLikelihood(sequence_length)
  test_clustering(d, 0.02, clusters)


def test_parameter_sampling(clusters):
  d = NaiveParameterSampling()
  test_clustering(d, 0.2, clusters)


def test_acgt_content(clusters):
  d = ACGTContent()
  test_clustering(d, 0.2, clusters)


def test_stationary_distribution(clusters):
  d = StationaryDistribution()
  test_clustering(d, 0.2, clusters)


def test_estimate_vlmc(sequence_length, clusters):
  inner_d = NegativeLogLikelihood(sequence_length)
  d = EstimateVLMC(inner_d)
  test_clustering(d, 0.2, clusters)


def test_clustering(d, threshold, clusters):
  tree_dir = "../trees"
  parse_trees_to_json.parse_trees(tree_dir)
  vlmcs = VLMC.from_json_dir(tree_dir)
  clustering = GraphBasedClustering(threshold, vlmcs, d)
  clustering.cluster(clusters)

if __name__ == '__main__':
  parser = argparse.ArgumentParser(
      description='Tests the distance functions for the vlmcs in ../trees, checking which vlmc they most closely match.')

  parser.add_argument('--parameter-sampling', action='store_true')
  parser.add_argument('--negative-log-likelihood', action='store_true')
  parser.add_argument('--acgt-content', action='store_true')
  parser.add_argument('--stationary-distribution', action='store_true')
  parser.add_argument('--estimate-vlmc', action='store_true')

  parser.add_argument('--seqlen', type=int, default=1000,
                      help='The length of the sequences that are generated to calculate the likelihood.')

  parser.add_argument('--clusters', type=int, default=10,
                      help='The number of clusters produced.')

  args = parser.parse_args()

  if (args.negative_log_likelihood):
    print('Testing negative log likelihood with a generated sequence of length {}'.format(args.seqlen))
    test_negloglike(args.seqlen, args.clusters)

  if (args.parameter_sampling):
    print('Testing the measure of estimation error distance function, the parameter based sampling.')
    test_parameter_sampling(args.clusters)

  if (args.acgt_content):
    print("Testing distance based only on acgt content.")
    test_acgt_content(args.clusters)

  if (args.stationary_distribution):
    print("Testing distance based on the stationary distribution")
    test_stationary_distribution(args.clusters)

  if (args.estimate_vlmc):
    print("Testing distance with an estimated vlmc")
    test_estimate_vlmc(args.seqlen, args.clusters)
