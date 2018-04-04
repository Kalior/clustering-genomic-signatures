#! /usr/bin/python3.6
from vlmc import VLMC
import parse_trees_to_json
from draw_vlmc import save, save_intersection
from test_distance_function import calculate_distances
from distance import FrobeniusNorm

import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

label_size = 20
mpl.rcParams['xtick.labelsize'] = label_size
mpl.rcParams['ytick.labelsize'] = label_size
mpl.rcParams['axes.axisbelow'] = True
mpl.rcParams['font.size'] = 24


def write_sequence_as_fasta(vlmc, sequence_length, out_directory):
  vlmc.reset_sequence()
  size_of_line = 990
  context = vlmc.generate_sequence(50, 500)[-vlmc.order:]

  file_name = vlmc.name + ".fa"
  with open(os.path.join(out_directory, file_name), 'w') as f:
    f.write("> {}\n".format(vlmc.name))
    # Iteratively generate lines from a sequence of correct length
    for _ in range(sequence_length // size_of_line):
      sequence = vlmc.generate_sequence_from(size_of_line, context)
      f.write("{}\n".format(sequence))
      context = sequence[-vlmc.order]

    length_left = sequence_length % size_of_line
    sequence = vlmc.generate_sequence_from(length_left, context)
    f.write("{}\n".format(sequence))


def add_underlines(directory):
  for file in [f for f in os.listdir(directory) if f.endswith(".tree")]:
    name, end = os.path.splitext(file)
    orignal_name = os.path.join(directory, file)
    new_name = os.path.join(directory, name + "__" + end)
    os.rename(orignal_name, new_name)


def generate(vlmcs, sequence_length, out_directory, number_of_parameters=128):
  os.system("rm {}/*".format(out_directory))

  for vlmc in vlmcs:
    write_sequence_as_fasta(vlmc, sequence_length, out_directory)

  list_path = os.path.join(out_directory, "list.txt")
  with open(list_path, 'a') as f:
    f.write('\n'.join([vlmc.name for vlmc in vlmcs]))

  parameters = {
      'use_constant_cutoff': "false",
      'cutoff_value': "3.9075",
      'number_of_parameters': number_of_parameters,
      'min_count': 4,
      'max_depth': 15
  }
  generate_vlmcs(vlmcs, parameters, list_path, out_directory)


def generate_vlmcs(vlmcs, parameters, list_path, out_directory):
  standard_args = "-pseudo -crr -f_f {} -ipf .fa -ipwd {} -opwd {} -osf TEST_ -m 1 -frac 0 -revcomp".format(
      list_path, out_directory, out_directory)
  parameter_args = "-c_c {} -nc {} -npar {} -minc {} -kmax {}".format(
      parameters['use_constant_cutoff'],
      parameters['cutoff_value'],
      parameters['number_of_parameters'],
      parameters['min_count'],
      parameters['max_depth']
  )

  args = ("../lib/classifier " + standard_args + " " + parameter_args).split()

  popen = subprocess.Popen(args, stdout=subprocess.PIPE)
  popen.wait()
  output = popen.stdout.read()
  # print(output.decode("utf-8"))
  # Needed for the parsing (since we expect them to be there...)
  add_underlines(out_directory)


def distance_calculation(pairs, d, results, i, repetitions):
  pair_distances = [d.distance(*p) for p in pairs]
  for j, distance in enumerate(pair_distances):
    results[i, j] += distance / repetitions


def pair_vlmcs(vlmcs, new_vlmcs):
  pairs = [(v1, v2) for v1 in vlmcs for v2 in new_vlmcs if v1.name in v2.name]
  return pairs


def calculate_distances_for_lengths(vlmcs, lengths, out_directory, image_directory):
  d = FrobeniusNorm()

  distances = np.zeros((len(lengths), len(vlmcs)))

  repetitions = 5
  for i, length in enumerate(lengths):
    print(length)
    for _ in range(repetitions):
      distance_repetitions(vlmcs, length, out_directory,
                           image_directory, distances, d, i, repetitions)

  return distances


def distance_repetitions_kl(vlmcs, length, out_directory, image_directory, distances, d, i, repetitions):
  pairs = [(v1, v2) for v1 in vlmcs for v2 in vlmcs if v1 != v2]
  plot_vlmcs(pairs, image_directory + "/" + str(i))
  distance_calculation(pairs, d, distances, i, repetitions)


def distance_repetitions(vlmcs, length, out_directory, image_directory, distances, d, i, repetitions):
  generate(vlmcs, length, out_directory)
  parse_trees_to_json.parse_trees(out_directory)
  new_vlmcs = VLMC.from_json_dir(out_directory)
  pairs = pair_vlmcs(vlmcs, new_vlmcs)
  plot_vlmcs(pairs, image_directory + "/" + str(length))
  distance_calculation(pairs, d, distances, i, repetitions)


def plot_vlmcs(pairs, image_directory):
  try:
    os.stat(image_directory)
  except:
    os.mkdir(image_directory)

  metadata = {v.name: {'species': v.name} for p in pairs for v in [*p]}
  for p in pairs:
    save_intersection([*p], metadata, image_directory)


def plot_results(results, image_directory):
  fig, ax = plt.subplots(1, figsize=(50, 30), dpi=80)

  ax.grid(color='#cccccc', linestyle='--', linewidth=1)
  ax.set_xticklabels(lengths)
  plt.xticks(np.arange(len(lengths)))

  handles = ax.plot(results, markersize=5, marker='o')

  labels = [v.name for v in vlmcs]
  plt.legend(handles=handles, labels=labels, bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)

  out_file = os.path.join(image_directory, 'distance-regeneration.pdf')
  plt.savefig(out_file, dpi='figure', format='pdf')

if __name__ == "__main__":
  out_directory = "../test_128"
  in_directory = "../test_trees_128"
  image_directory = "../images/128"

  parse_trees_to_json.parse_trees(in_directory)
  vlmcs = VLMC.from_json_dir(in_directory)

  lengths = [int(l) for l in np.logspace(2, 6, 10)]
  print(lengths)
  distances = calculate_distances_for_lengths(vlmcs, lengths, out_directory, image_directory)
  plot_results(distances, image_directory)
