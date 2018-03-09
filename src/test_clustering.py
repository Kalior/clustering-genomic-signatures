#! /usr/bin/python3.6
import argparse
import time
import matplotlib.pyplot as plt
import networkx as nx

from vlmc import VLMC
from distance import NegativeLogLikelihood, NaiveParameterSampling, StationaryDistribution, ACGTContent, FrobeniusNorm, EstimateVLMC
from clustering import GraphBasedClustering, MSTClustering, MinInterClusterDistance
import parse_trees_to_json
from get_signature_metadata import get_metadata_for


def test_negloglike(sequence_length, clusters):
  d = NegativeLogLikelihood(sequence_length)
  test_clustering(d, clusters)


def test_parameter_sampling(clusters):
  d = NaiveParameterSampling()
  test_clustering(d, clusters)


def test_acgt_content(clusters):
  d = ACGTContent()
  test_clustering(d, clusters)


def test_stationary_distribution(clusters):
  d = StationaryDistribution()
  test_clustering(d, clusters)


def test_estimate_vlmc(sequence_length, clusters):
  inner_d = NegativeLogLikelihood(sequence_length)
  d = EstimateVLMC(inner_d)
  test_clustering(d, clusters)


def test_frobenius(clusters):
  d = FrobeniusNorm()
  test_clustering(d, clusters)


def test_estimate_vlmc(sequence_length, clusters):
  inner_d = FrobeniusNorm()
  d = EstimateVLMC(inner_d)
  test_clustering(d, clusters)


def test_clustering(d, clusters, draw_graph=False):
  tree_dir = "../trees"
  parse_trees_to_json.parse_trees(tree_dir)
  vlmcs = VLMC.from_json_dir(tree_dir)
  clustering = MinInterClusterDistance(vlmcs, d)
  G, distance_mean = clustering.cluster(clusters)

  if draw_graph:
    draw_graph(G)
  print_connected_components(G, vlmcs, d, distance_mean)


def draw_graph(self, G):
  plt.subplot(121)
  nx.draw_shell(G, with_labels=True, font_weight='bold')
  plt.show()


def print_connected_components(G, vlmcs, d, distance_mean):
  metadata = get_metadata_for([vlmc.name for vlmc in vlmcs])

  connected_component_metrics = [component_metrics(
      connected, metadata, d) for connected in nx.connected_components(G)]

  output = ["cluster {}:\n".format(i) + component_string(connected, metadata, connected_component_metrics[i])
            for i, connected in enumerate(nx.connected_components(G))]

  print('\n\n'.join(output))

  filtered_metrics = [metrics for metrics, connected in zip(
      connected_component_metrics, nx.connected_components(G)) if len(connected) > 1]

  average_of_same_genus = sum(
      [metrics[0] for metrics in filtered_metrics]) / len(filtered_metrics)
  average_of_same_family = sum(
      [metrics[1] for metrics in filtered_metrics]) / len(filtered_metrics)
  total_average_distance = sum(
      [metrics[2] for metrics in filtered_metrics]) / len(filtered_metrics) / distance_mean

  print("Average percent of same genus in clusters: {:5.5f}\t"
        "Average percent of same family in clusters: {:5.5f}\t"
        "Average distance in clusters: {:5.5f}\t".format(
            average_of_same_genus, average_of_same_family, total_average_distance))

  sorted_sizes = sorted([len(connected) for connected in nx.connected_components(G)])
  print("Cluster sizes " + " ".join([str(i) for i in sorted_sizes]))


def component_metrics(connected, metadata, d):
  percent_of_same_genus = sum(
      [number_in_taxonomy(vlmc, connected, metadata, 'genus') for vlmc in connected]
  ) / (len(connected) * len(connected))

  percent_of_same_family = sum(
      [number_in_taxonomy(vlmc, connected, metadata, 'family') for vlmc in connected]
  ) / (len(connected) * len(connected))

  connected_distances = [d.distance(v1, v2) for v1 in connected for v2 in connected]
  average_distance = sum(connected_distances) / len(connected_distances)

  return percent_of_same_genus, percent_of_same_family, average_distance


def component_string(connected, metadata, metrics):
  output = [output_line(metadata, vlmc) for vlmc in connected]

  metric_string = "\nPercent of same genus: {:5.5f} \t Percent of same family: {:5.5f} \t Average distance: {:5.5f}\n".format(
      metrics[0], metrics[1], metrics[2])

  return '\n'.join(output) + metric_string


def output_line(metadata, vlmc):
  return "{:>55}  {:20} {:20}".format(
      metadata[vlmc.name]['species'],
      metadata[vlmc.name]['genus'],
      metadata[vlmc.name]['family'])


def number_in_taxonomy(vlmc, vlmcs, metadata, taxonomy):
  number_of_same_taxonomy = len([other for other in vlmcs
                                 if metadata[other.name][taxonomy] == metadata[vlmc.name][taxonomy]])
  return number_of_same_taxonomy


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
      description='Tests the distance functions for the vlmcs in ../trees, checking which vlmc they most closely match.')

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

  if (args.frobenius_norm):
    print("Testing clustering with distance as frobenius norm")
    test_frobenius(args.clusters)
