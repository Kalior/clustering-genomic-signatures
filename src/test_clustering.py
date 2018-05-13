#! /usr/bin/python3.6
import argparse
import time
import os

import matplotlib as mpl

if __name__ == '__main__':
  label_size = 20 * 2
  mpl.rcParams['xtick.labelsize'] = label_size
  mpl.rcParams['ytick.labelsize'] = label_size
  mpl.rcParams['axes.axisbelow'] = True
  mpl.rcParams['font.size'] = 24 * 2

from vlmc import VLMC
from clustering import *
import parse_trees_to_json
from get_signature_metadata import get_metadata_for
from test_distance_function import parse_distance_method, add_distance_arguments
from util.draw_clusters import draw_graph, plot_largest_components
from util.print_clusters import print_connected_components


def test_clustering(d, clusters, vlmcs, out_directory, cluster_class=MSTClustering, do_draw_graph=True):
  metadata = get_metadata_for([vlmc.name for vlmc in vlmcs])

  clustering = cluster_class(vlmcs, d, metadata)
  for i in range(clusters + 0, clusters - 1, -1):
    print(i)
    clustering_metrics = clustering.cluster(i)

    if do_draw_graph:
      plot_largest_components(clustering_metrics, i, out_directory)

      pictures = [('Family', 'family'), ('Genus', 'genus'),
                  ('Host', 'hosts'), ('Baltimore', 'baltimore')]
      for name, key in pictures:
        draw_graph(clustering_metrics, name, key, i, out_directory)

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
  elif args.dendrogram:
    print("Clustering to dendrogram with the help of scipy")
    return DendrogramClustering
  else:
    print("Clustering with min single linkage")
    return AverageLinkClustering


def test(args):
  vlmcs = parse_trees(args)
  cluster_class = parse_clustering_method(args)
  d = parse_distance_method(args)

  try:
    os.stat(args.out_directory)
  except:
    os.mkdir(args.out_directory)

  test_clustering(d, args.clusters, vlmcs, args.out_directory, cluster_class, args.draw_graph)


def add_clustering_arguments(parser):
  parser.add_argument('--average-link-clustering', action='store_true')
  parser.add_argument('--single-link-clustering', action='store_true')
  parser.add_argument('--fuzzy-similarity-clustering', action='store_true')
  parser.add_argument('--kmeans', action='store_true')
  parser.add_argument('--dendrogram', action='store_true')

if __name__ == '__main__':
  parser = argparse.ArgumentParser(
      description=('Tests the clustering/distance functions for vlmcs,'
                   'checking which vlmc they most closely match.'))

  add_distance_arguments(parser)
  add_clustering_arguments(parser)

  parser.add_argument('--clusters', type=int, default=10,
                      help='The number of clusters produced.')

  parser.add_argument('--directory', type=str, default='../trees',
                      help='The directory to source the trees for the VLMCs from.')
  parser.add_argument('--out-directory', type=str, default='../images',
                      help='The directory to where images are written.')
  parser.add_argument('--draw-graph', action='store_true')

  args = parser.parse_args()
  test(args)
