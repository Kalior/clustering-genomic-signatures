#! /usr/bin/python3.6
import argparse
import time

from vlmc import VLMC
from distance import *
from clustering import *
import parse_trees_to_json
from get_signature_metadata import get_metadata_for
from util.draw_clusters import draw_graph, draw_silhouette
from util.print_clusters import print_connected_components


def test_negloglike(sequence_length, clusters, cluster_class, vlmcs):
  d = NegativeLogLikelihood(sequence_length)
  test_clustering(d, clusters, vlmcs, cluster_class)


def test_parameter_sampling(clusters, cluster_class, vlmcs):
  d = NaiveParameterSampling()
  test_clustering(d, clusters, vlmcs, cluster_class)


def test_acgt_content(clusters, cluster_class, vlmcs):
  d = ACGTContent()
  test_clustering(d, clusters, vlmcs, cluster_class)


def test_stationary_distribution(clusters, cluster_class, vlmcs):
  d = StationaryDistribution()
  test_clustering(d, vlmcs, cluster_class, clusters)


def test_frobenius(clusters, cluster_class, vlmcs):
  d = FrobeniusNorm()
  test_clustering(d, clusters, vlmcs, cluster_class)


def test_estimate_vlmc(sequence_length, clusters, cluster_class, vlmcs):
  inner_d = FrobeniusNorm()
  d = EstimateVLMC(inner_d)
  test_clustering(d, clusters, vlmcs, cluster_class)


def test_kmeans(k, vlmcs):
  d = Projection(vlmcs)
  test_clustering(d, k, vlmcs, cluster_class=KMeans)


def test_clustering(d, clusters, vlmcs, cluster_class=MSTClustering, do_draw_graph=True):
  clustering = cluster_class(vlmcs, d)
  clustering_metrics = clustering.cluster(clusters)
  metadata = get_metadata_for([vlmc.name for vlmc in vlmcs])
  clustering_metrics.metadata = metadata

  if do_draw_graph:
    draw_graph(clustering_metrics.G, metadata)
    draw_silhouette(clustering_metrics)
  print_connected_components(clustering_metrics)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
      description='Tests the clustering/distance functions for vlmcs, checking which vlmc they most closely match.')

  parser.add_argument('--parameter-sampling', action='store_true')
  parser.add_argument('--negative-log-likelihood', action='store_true')
  parser.add_argument('--acgt-content', action='store_true')
  parser.add_argument('--stationary-distribution', action='store_true')
  parser.add_argument('--estimate-vlmc', action='store_true')
  parser.add_argument('--frobenius-norm', action='store_true')

  parser.add_argument('--seqlen', type=int, default=1000,
                      help='The length of the sequences that are generated to calculate the likelihood.')

  parser.add_argument('--clusters', type=int, default=10,
                      help='The number of clusters produced.')

  parser.add_argument('--directory', type=str, default='../trees',
                      help='The directory to source the trees for the VLMCs from.')

  parser.add_argument('--average-link-clustering', action='store_true')
  parser.add_argument('--single-link-clustering', action='store_true')
  parser.add_argument('--fuzzy-similarity-clustering', action='store_true')
  parser.add_argument('--kmeans', action='store_true')

  args = parser.parse_args()
  tree_dir = args.directory
  parse_trees_to_json.parse_trees(tree_dir)
  vlmcs = VLMC.from_json_dir(tree_dir)

  if args.average_link_clustering:
    print("Clustering with min average distance between clusters")
    cluster_class = AverageLinkClustering
  elif args.single_link_clustering:
    print("Clustering with min single linkage")
    cluster_class = MSTClustering
  elif args.fuzzy_similarity_clustering:
    print("Clustering with the fuzzy similarity measure")
    cluster_class = FuzzySimilarityClustering
  else:
    print("Clustering with min single linkage")
    cluster_class = MSTClustering

  if (args.negative_log_likelihood):
    print('Testing negative log likelihood with a generated sequence of length {}'.format(args.seqlen))
    test_negloglike(args.seqlen, args.clusters, cluster_class, vlmcs)

  if (args.parameter_sampling):
    print('Testing the measure of estimation error distance function, the parameter based sampling.')
    test_parameter_sampling(args.clusters, cluster_class, vlmcs)

  if (args.acgt_content):
    print("Testing distance based only on acgt content.")
    test_acgt_content(args.clusters, cluster_class, vlmcs)

  if (args.stationary_distribution):
    print("Testing distance based on the stationary distribution")
    test_stationary_distribution(args.clusters, cluster_class, vlmcs)

  if (args.estimate_vlmc):
    print("Testing distance with an estimated vlmc")
    test_estimate_vlmc(args.seqlen, args.clusters, cluster_class, vlmcs)

  if (args.frobenius_norm):
    print("Testing clustering with distance as frobenius norm")
    test_frobenius(args.clusters, cluster_class, vlmcs)

  if args.kmeans:
    print("Testing k means clustering with k = {}".format(args.clusters))
    test_kmeans(args.clusters, vlmcs)
