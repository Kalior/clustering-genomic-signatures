#! /usr/bin/python3.6
import os
import subprocess
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import argparse

label_size = 20
mpl.rcParams['xtick.labelsize'] = label_size
mpl.rcParams['ytick.labelsize'] = label_size
mpl.rcParams['axes.axisbelow'] = True
mpl.rcParams['font.size'] = 24

from vlmc import VLMC
from distance import FrobeniusNorm, NegativeLogLikelihood
from train import train_vlmcs
from parse_trees_to_json import parse_trees


def test(args):
  virus_aids = [
      "JQ596859.1",
      "KF487736.1",
      "KJ627438.1",
      "NC_001493.2",
      "NC_008725.1",
      "NC_009011.2",
      "KF234407.1"
  ]
  list_path = "list.txt"
  out_directory = "../test_increasing"
  image_directory = os.path.join(out_directory, "images")

  for d in [out_directory, image_directory]:
    try:
      os.stat(d)
    except:
      os.mkdir(d)

  free_parameters = np.arange(args.start, args.end, args.step_size)
  if not args.use_existing_models:
    write_fasta_files(virus_aids)
    write_list(list_path, virus_aids)
    train_with_free_parameters(free_parameters, list_path, out_directory)

  if args.iterative_comparison:
    distances = distances_between_parameters(free_parameters, out_directory, len(virus_aids))
  elif args.fixed_comparison:
    distances = distances_to(free_parameters, out_directory, len(virus_aids), args.compare_to)

  plot_distances(distances, virus_aids, free_parameters, image_directory)


def write_fasta_files(aids):
  os.system("mkdir -p {}".format("fasta"))

  os.system("rm -f fasta/*")

  select = "(" + " OR ".join(["virus.aid='{}'".format(v) for v in aids]) + ")"
  args = ["perl", "../lib/db2fasta.pl", "-c", select, "-e", "-p", "100"]

  db2_fasta = subprocess.Popen(args, stdout=subprocess.PIPE)
  db2_fasta.wait()


def write_list(list_path, aids):
  os.system("rm -f {}".format(list_path))
  os.system("ls -1 fasta/*.fa | /usr/bin/sed 's!.*/!!' | /usr/bin/sed 's/\.fa$//' > {}".format(list_path))


def train_with_free_parameters(free_parameters, list_path, out_directory):
  for number_of_parameters in free_parameters:
    parameter_directory = os.path.join(out_directory, str(number_of_parameters))
    train_with(number_of_parameters, list_path, parameter_directory)


def train_with(number_of_parameters, list_path, out_directory):
  try:
    os.stat(out_directory)
    os.system("rm -rf {}/*".format(out_directory))
  except:
    os.system("mkdir -p {}".format(out_directory))

  parameters = {
      'use_constant_cutoff': "false",
      'cutoff_value': "3.9075",
      'number_of_parameters': number_of_parameters,
      'min_count': 4,
      'max_depth': 15,
      'count_free_parameters_individually': 'true'
  }
  train_vlmcs(parameters, list_path, out_directory, "fasta", False)


def distances_between_parameters(free_parameters, out_directory, number_of_vlmcs):
  d = FrobeniusNorm()

  distances = np.zeros([len(free_parameters) - 1, number_of_vlmcs])

  for i in range(1, len(free_parameters)):
    first_vlmcs = get_vlmcs(out_directory, free_parameters[i - 1])
    second_vlmcs = get_vlmcs(out_directory, free_parameters[i])

    pairs = pair(first_vlmcs, second_vlmcs)
    results = np.array([d.distance(*p) for p in pairs])
    distances[i - 1, :] = results

  return distances


def distances_to(free_parameters, out_directory, number_of_vlmcs, fixed_parameter):
  d = FrobeniusNorm()

  distances = np.empty([len(free_parameters), number_of_vlmcs])

  fixed_vlmcs = get_vlmcs(out_directory, fixed_parameter)

  with multiprocessing.Pool(processes=3) as pool:
    results = [pool.apply_async(distance_calculation,
                                (fixed_vlmcs, out_directory, number_of_parameters, d))
               for i, number_of_parameters in enumerate(free_parameters)]

    distances = np.stack([res.get() for res in results])

  return distances


def distance_calculation(fixed_vlmcs, out_directory, number_of_parameters, d):
  vlmcs = get_vlmcs(out_directory, number_of_parameters)

  pairs = pair(fixed_vlmcs, vlmcs)
  results = np.array([d.distance(*p) for p in pairs])
  return results


def get_vlmcs(out_directory, number_of_parameters):
  directory = os.path.join(out_directory, str(number_of_parameters))
  parse_trees(directory)
  return VLMC.from_json_dir(directory)


def pair(first_vlmcs, second_vlmcs):
  return [(v1, v2) for v1 in first_vlmcs for v2 in second_vlmcs if v1.name == v2.name]


def plot_distances(distances, virus_aids, free_parameters, image_directory):
  fig, ax = plt.subplots(1, figsize=(50, 30), dpi=80)

  ax.grid(color='#cccccc', linestyle='--', linewidth=1)

  xticks_step = (len(free_parameters) // 20) + 1
  xtick_locs = np.arange(0, len(free_parameters), xticks_step)

  ax.set_xticks(xtick_locs)
  ax.set_xticklabels(free_parameters[xtick_locs])

  ax.set_title('Increasing parameters distance')

  handles = ax.plot(distances, markersize=5, marker='o')

  ax.legend(handles=handles, labels=virus_aids, bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)

  out_file = os.path.join(image_directory, 'increasing-parameters-distance.pdf')
  plt.savefig(out_file, dpi='figure', format='pdf')


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
      description='Tests models with different number of parameters to each other.')
  parser.add_argument('--fixed-comparison', action='store_true',
                      help='Compare every other model to a fixed number of parmeters.')
  parser.add_argument('--compare-to', type=int, default=48,
                      help='The number of parameters to compare each model to. Needed for fixed comparison.')

  parser.add_argument('--iterative-comparison', action='store_true',
                      help='Compare every model to every next model (#steps more parameters).')

  parser.add_argument('--start', type=int, default=24,
                      help='The number of parameters for the first model.')
  parser.add_argument('--end', type=int, default=192,
                      help='The number of parameters for the last model.')
  parser.add_argument('--step-size', type=int, default=1,
                      help='The step size between each subsequent pair of models.')

  parser.add_argument('--use-existing-models', action='store_true',
                      help='Does not retrain the models, assuming they already exist, saves some time.')

  args = parser.parse_args()
  test(args)
