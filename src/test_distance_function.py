#! /usr/bin/python3.6
from vlmc import VLMC
from distance import NegativeLogLikelihood, NaiveParameterSampling, StationaryDistribution, ACGTContent, FrobeniusNorm
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


def test_frobenius_norm(tree_dir):
  d = FrobeniusNorm()
  test_distance_function(d, tree_dir)


def test_distance_function(d, tree_dir):
  parse_trees_to_json.parse_trees(tree_dir)
  vlmcs = VLMC.from_json_dir(tree_dir)
  metadata = get_metadata_for([vlmc.name for vlmc in vlmcs])

  average_procent_of_genus_in_top = 0.0
  average_procent_of_family_in_top = 0.0
  total_average_distance_to_genus = 0.0
  total_average_distance_to_family = 0.0
  total_average_distance = 0.0

  for vlmc in vlmcs:
    start_time = time.time()
    distances = list(map(lambda other: d.distance(vlmc, other), vlmcs))
    elapsed_time = time.time() - start_time

    sorted_results = sorted(zip(distances, vlmcs),
                            key=lambda t: (t[0], metadata[t[1].name]['genus']))

    procent_genus_in_top = procent_of_taxonomy_in_top(
        vlmc, vlmcs, sorted_results, metadata, 'genus')
    procent_family_in_top = procent_of_taxonomy_in_top(
        vlmc, vlmcs, sorted_results, metadata, 'family')

    average_distance_to_genus = average_distance_to_taxonomy(
        vlmc, sorted_results, metadata, 'genus')
    average_distance_to_family = average_distance_to_taxonomy(
        vlmc, sorted_results, metadata, 'family')
    average_distance = sum(distances) / len(distances)

    # test_output(vlmc, vlmcs, sorted_results, elapsed_time, metadata,
    #             procent_genus_in_top, procent_family_in_top,
    #             average_distance_to_genus, average_distance_to_family, average_distance)

    average_procent_of_genus_in_top += procent_genus_in_top
    average_procent_of_family_in_top += procent_family_in_top
    total_average_distance_to_genus += average_distance_to_genus
    total_average_distance_to_family += average_distance_to_family
    total_average_distance += average_distance

  average_procent_of_genus_in_top /= len(vlmcs)
  average_procent_of_family_in_top /= len(vlmcs)
  total_average_distance /= len(vlmcs)
  total_average_distance_to_genus /= (len(vlmcs) * total_average_distance)
  total_average_distance_to_family /= (len(vlmcs) * total_average_distance)
  print("Average procent of genus in top #genus: {:5.5f} \t Average procent of family in top #family {:5.5f}\n"
        "Average distance fraction to genus: {:5.5f} \t Average distance fraction to family {:5.5f} \t Average distance: {:5.5f}\n".format(
            average_procent_of_genus_in_top, average_procent_of_family_in_top,
            total_average_distance_to_genus, total_average_distance_to_family, total_average_distance))


def test_output(vlmc, vlmcs, sorted_results, elapsed_time, metadata, procent_genus_in_top, procent_family_in_top, distance_to_genus, distance_to_family, average_distance):
  print("{}:".format(metadata[vlmc.name]['species']))
  print("Procent of genus in top #genus: {:5.5f} \t Procent of family in top #family {:5.5f}\n"
        "Average distance to genus: {:5.5f} \t Average distance to family {:5.5f} \t Average distance: {:5.5f}".format(
            procent_genus_in_top, procent_family_in_top, distance_to_genus, distance_to_family, average_distance))
  print("matches self: {}.\tDistance calculated in: {}s\n".format(
      vlmc == sorted_results[0][1], elapsed_time))

  extra_distance = ACGTContent(['C', 'G'])
  result_list = [output_line(metadata, vlmc, dist, v, extra_distance)
                 for (dist, v) in sorted_results]

  print('\n'.join(result_list) + '\n\n')


def procent_of_taxonomy_in_top(vlmc, vlmcs, sorted_results, metadata, taxonomy):
  number_same_taxonomy = len([other for other in vlmcs
                              if metadata[other.name][taxonomy] == metadata[vlmc.name][taxonomy]])

  number_same_in_top = 0
  for i in range(number_same_taxonomy):
    _, v = sorted_results[i]
    if metadata[v.name][taxonomy] == metadata[vlmc.name][taxonomy]:
      number_same_in_top += 1

  return number_same_in_top / number_same_taxonomy


def average_distance_to_taxonomy(vlmc, sorted_results, metadata, taxonomy):
  same_taxonomy = [dist for (dist, other) in sorted_results
                   if metadata[other.name][taxonomy] == metadata[vlmc.name][taxonomy]]
  return sum(same_taxonomy) / len(same_taxonomy)


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
  parser.add_argument('--frobenius-norm', action='store_true')

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
    print("Testing distance based on the stationary distribution.")
    test_stationary_distribution(args.directory)

  if (args.frobenius_norm):
    print("Testing distance frobenius norm.")
    test_frobenius_norm(args.directory)
