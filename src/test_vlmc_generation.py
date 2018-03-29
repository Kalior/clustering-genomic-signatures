from vlmc import VLMC
import parse_trees_to_json
from draw_vlmc import save
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


def generate(vlmcs, sequence_length, list_path, out_directory):
  os.system("rm {}/*".format(out_directory))
  sequences = [(vlmc.generate_sequence(sequence_length, 500), vlmc) for vlmc in vlmcs]

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
  # ./classifier -pseudo -crr -f_f list.txt -ipf .fa -ipwd test -opwd profiles -osf PST_ -m 1 -frac 0 -revcomp -npar 768 -minc 40 -kmax 9 | tee -a $OUTPUT
  # args = "bin/bar -c somefile.xml -d text.txt -r aString -f anotherString".split()
  popen = subprocess.Popen(args, stdout=subprocess.PIPE)
  popen.wait()
  output = popen.stdout.read()
  # print(output.decode("utf-8"))
  # Needed for the parsing (since we expect them to be there...)
  add_underlines(out_directory)


def distance_calculation(vlmcs, new_vlmcs, d, results, i):
  pairs = pair_vlmcs(vlmcs, new_vlmcs)
  pair_distances = [d.distance(*p) for p in pairs]
  for j, distance in enumerate(pair_distances):
    results[i, j] = distance


def pair_vlmcs(vlmcs, new_vlmcs):
  pairs = [(v1, v2) for v1 in vlmcs for v2 in new_vlmcs if v1.name in v2.name]
  return pairs

if __name__ == "__main__":
  out_directory = "../test"
  in_directory = "../trees_128"
  image_directory = "../images"
  list_path = os.path.join(out_directory, "list.txt")

  d = FrobeniusNorm()

  fig, ax = plt.subplots(1, figsize=(150, 30), dpi=80)
  ax.grid(color='#cccccc', linestyle='--', linewidth=1)

  parse_trees_to_json.parse_trees(in_directory)
  vlmcs = VLMC.from_json_dir(in_directory)
  lengths = np.concatenate((np.arange(1000, 500000, 10000), np.array([100000, 500000, 1000000])))

  results = np.empty((len(lengths), len(vlmcs)))
  for i, length in enumerate(lengths):
    generate(vlmcs, length, list_path, out_directory)
    parse_trees_to_json.parse_trees(out_directory)
    new_vlmcs = VLMC.from_json_dir(out_directory)
    distance_calculation(vlmcs, new_vlmcs, d, results, i)

  ax.set_xticklabels(lengths)
  plt.xticks(np.arange(len(lengths)))
  handles = ax.plot(results, markersize=5, marker='o')
  labels = [v.name for v in vlmcs]
  ax.legend(handles=handles, labels=labels, loc='upper right')

  out_file = os.path.join(image_directory, 'distance-regeneration.pdf')
  plt.savefig(out_file, dpi='figure', format='pdf', bbox_inches='tight')

  plt.show()
