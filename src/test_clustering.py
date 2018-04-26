#! /usr/bin/python3.6
import argparse
import time
import os

from vlmc import VLMC
from clustering import *
import parse_trees_to_json
from get_signature_metadata import get_metadata_for
from test_distance_function import parse_distance_method
from util.draw_clusters import draw_graph
from util.print_clusters import print_connected_components


def test_clustering(d, clusters, vlmcs, out_directory, cluster_class=MSTClustering, do_draw_graph=True):
  metadata = get_metadata_for([vlmc.name for vlmc in vlmcs])

  clustering = cluster_class(vlmcs, d)
  for i in range(clusters + 0, clusters - 1, -1):
    print(i)
    clustering_metrics = clustering.cluster(i)
    clustering_metrics.metadata = metadata

    if do_draw_graph:
      draw_graph(clustering_metrics, 'Family', 'family', i, out_directory)
    print_connected_components(clustering_metrics)


def parse_trees(args):
  tree_dir = args.directory
  parse_trees_to_json.parse_trees(tree_dir)
  return VLMC.from_json_dir(tree_dir)


def parse_clustering_method(args):
  if args.average_link_clustering:
    print("Clustering with min average distance between clusters")
    return AverageLinkClustering
  elif args.single_link_clustering:
    print("Clustering with min single linkage")
    return MSTClustering
  elif args.fuzzy_similarity_clustering:
    print("Clustering with the fuzzy similarity measure")
    return FuzzySimilarityClustering
  elif args.kmeans:
    print("Testing k means clustering with k = {}".format(args.clusters))
    return KMeans
  else:
    print("Clustering with min single linkage")
    return MSTClustering


def test(args):
  vlmcs = parse_trees(args)
  cluster_class = parse_clustering_method(args)
  d = parse_distance_method(args)

  try:
    os.stat(args.out_directory)
  except:
    os.mkdir(args.out_directory)

  test_clustering(d, args.clusters, vlmcs, args.out_directory, cluster_class, do_draw_graph=True)


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
  parser.add_argument('--out-directory', type=str, default='../images',
                      help='The directory to where images are written.')

  parser.add_argument('--average-link-clustering', action='store_true')
  parser.add_argument('--single-link-clustering', action='store_true')
  parser.add_argument('--fuzzy-similarity-clustering', action='store_true')
  parser.add_argument('--kmeans', action='store_true')

  args = parser.parse_args()
  test(args)
