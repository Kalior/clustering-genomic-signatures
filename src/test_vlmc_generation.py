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


def chunks(l, n):
  """Yield successive n-sized chunks from l."""
  for i in range(0, len(l), n):
    yield l[i:i + n]


def write_sequence_as_fasta(vlmc_tuple, out_directory, list_path):
  (seq, vlmc) = vlmc_tuple
  file_name = vlmc.name + ".fa"
  with open(os.path.join(out_directory, file_name), 'w') as f:
    f.write("> {}\n".format(vlmc.name))
    f.write('\n'.join(list(chunks(seq, 990))))

  with open(list_path, 'a') as f:
    f.write(vlmc.name + "\n")


def add_underlines(directory):
  for file in [f for f in os.listdir(directory) if f.endswith(".tree")]:
    name, end = os.path.splitext(file)
    orignal_name = os.path.join(directory, file)
    new_name = os.path.join(directory, name + "__" + end)
    os.rename(orignal_name, new_name)


def generate_sequence(vlmc, sequence_length):
  vlmc.reset_sequence()
  sequence = vlmc.generate_sequence(sequence_length, 500)
  return sequence, vlmc


def generate(vlmcs, sequence_length, out_directory):
  os.system("rm {}/*".format(out_directory))
  list_path = os.path.join(out_directory, "list.txt")

  sequences = [generate_sequence(vlmc, sequence_length) for vlmc in vlmcs]

  for seq in sequences:
    write_sequence_as_fasta(seq, out_directory, list_path)

  parameters = {
      'use_constant_cutoff': "false",
      'cutoff_value': "3.9075",
      'number_of_parameters': 48,
      'min_count': 4,
      'max_depth': 10
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


def distance_calculation(pairs, d, results, i):
  pair_distances = [d.distance(*p) for p in pairs]
  for j, distance in enumerate(pair_distances):
    results[i, j] = distance


def pair_vlmcs(vlmcs, new_vlmcs):
  pairs = [(v1, v2) for v1 in vlmcs for v2 in new_vlmcs if v1.name in v2.name]
  return pairs


def calculate_distances_for_lengths(vlmcs, lengths, out_directory, image_directory):
  d = FrobeniusNorm()

  distances = np.empty((len(lengths), len(vlmcs)))

  for i, length in enumerate(lengths):
    generate(vlmcs, length, out_directory)
    parse_trees_to_json.parse_trees(out_directory)
    new_vlmcs = VLMC.from_json_dir(out_directory)
    pairs = pair_vlmcs(vlmcs, new_vlmcs)
    plot_vlmcs(pairs, image_directory + "/" + str(i))
    distance_calculation(pairs, d, distances, i)

  return distances


def plot_vlmcs(pairs, image_directory):
  try:
    os.stat(image_directory)
  except:
    os.mkdir(image_directory)

  metadata = {v.name: {'species': v.name} for p in pairs for v in [*p]}
  for p in pairs:
    save_intersection([*p], metadata, image_directory)


def plot_results(results, image_directory):
  fig, ax = plt.subplots(1, figsize=(150, 30), dpi=80)

  ax.grid(color='#cccccc', linestyle='--', linewidth=1)
  ax.set_xticklabels(lengths)
  plt.xticks(np.arange(len(lengths), 5))

  handles = ax.plot(results, markersize=5, marker='o')

  labels = [v.name for v in vlmcs]
  plt.legend(handles=handles, labels=labels, bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)

  out_file = os.path.join(image_directory, 'distance-regeneration.pdf')
  plt.savefig(out_file, dpi='figure', format='pdf', bbox_inches='tight')

if __name__ == "__main__":
  out_directory = "../test"
  in_directory = "../test_trees"
  image_directory = "../images"

  parse_trees_to_json.parse_trees(in_directory)
  vlmcs = VLMC.from_json_dir(in_directory)

  lengths = np.concatenate((np.arange(5000, 50000, 5000), np.array([100000, 500000, 1000000])))
  distances = calculate_distances_for_lengths(vlmcs, lengths, out_directory, image_directory)
  plot_results(distances, image_directory)
