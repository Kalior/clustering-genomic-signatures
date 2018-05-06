#!/usr/bin/python3.6
from train import train_vlmcs
from test_distance_function import test_distance_function
from vlmc import VLMC
from distance import FrobeniusNorm
import numpy as np
import os
import subprocess
import sys
import matplotlib.pyplot as plt


# k_values is all different "orders" to test for the full order markov chain
# test vlmcs in range(min_param, max_param, step=step_size)
def main(d=FrobeniusNorm(), use_small_data_set=True, min_param=1000, max_param=10000, step_size=1000, k_values=[3, 4, 5, 6, 7, 8], directory_trained_models="../lib/trainedmodels", min_count=4, max_depth=15, from_start='f', percentage=100):
  """Given range of parameters, 30 to 1000 generate VLMCs of every 10ish
parameter size use test_distance to calculate metrics for the distance
function given"""

  # start by extracting the fasta files into "../lib/fasta"-directory
  # assume db2fasta lies in working directory "."
  file_of_vlmc_names = "/tmp/list.txt" # had trouble using local file
  tree_files_exists = True
  if not tree_files_exists:
    extract_from_data_set_fasta_files(from_start, percentage, use_small_data_set=use_small_data_set)
  os.system("ls -1 ../lib/fasta/*.fa | /bin/sed 's!.*/!!' | /bin/sed 's/\.fa$//' > {}".format(file_of_vlmc_names))

  parameters = {
      'use_constant_cutoff': "false",
      'cutoff_value': "3.9075",
      'number_of_parameters': 0,
      'min_count': min_count,
      'max_depth': max_depth,
      'count_free_parameters_individually': "false", # false = Dalevi's way of calculating the nbr of parameters
      'generate_full_markov_chain': False,
      'markov_chain_order': 3
  }

  all_parameters_to_test = sorted([1000, 107, 137, 167, 17, 197, 2000, 220, 3000, 320, 4000, 420, 47, 5000, 520, 6000, 620, 7000, 720, 77, 8000, 820, 9000, 920, 40000, 100000])
  nbr_data_points = len(all_parameters_to_test)

  metrics_data_vlmc = {
      "average_procent_of_genus_in_top": np.zeros(nbr_data_points),
      "average_procent_of_family_in_top": np.zeros(nbr_data_points),
      "total_average_distance_to_genus": np.zeros(nbr_data_points),
      "total_average_distance_to_family": np.zeros(nbr_data_points),
      "total_average_distance": np.zeros(nbr_data_points),
      "global_time": np.zeros(nbr_data_points)
  }
  
  for datum_index, nbr_param in enumerate(all_parameters_to_test):
    parameters['number_of_parameters'] = nbr_param
    param_directory = directory_trained_models + "/vlmc_{}_parameters".format(nbr_param)
    if not os.path.exists(param_directory):
      # if we haven't already generated these, models, generate them
      os.makedirs(param_directory)
      train_vlmcs(parameters, file_of_vlmc_names, param_directory, input_directory="/home/neon/Documents/exjobb/clustering-genomic-signatures/lib/fasta", add_underlines_=False)
    metrics = test_distance_function(d, param_directory, out_dir=None)
    print(metrics)
    for key in metrics_data_vlmc.keys():
      metrics_data_vlmc[key][datum_index] = metrics[key]
  plt.figure(1)
  for k, v in metrics_data_vlmc.items():
    plt.semilogx(all_parameters_to_test, v, label=k, marker='.')
  plt.legend()
  plt.grid(True)
  plt.show()
  
  nbr_data_points = len(k_values)
  metrics_data_markov_chain = {
      "average_procent_of_genus_in_top": np.zeros(nbr_data_points),
      "average_procent_of_family_in_top": np.zeros(nbr_data_points),
      "total_average_distance_to_genus": np.zeros(nbr_data_points),
      "total_average_distance_to_family": np.zeros(nbr_data_points),
      "total_average_distance": np.zeros(nbr_data_points),
      "global_time": np.zeros(nbr_data_points)
  }
      
  parameters['generate_full_markov_chain'] = True
  for datum_index, order in enumerate(k_values):
    parameters['markov_chain_order'] = order
    param_directory = directory_trained_models + "/mc_{}_parameters".format(order)
    if not os.path.exists(param_directory):
      # if we haven't already generated these, models, generate them
      os.makedirs(param_directory)
      train_vlmcs(parameters, file_of_vlmc_names, param_directory, input_directory="/home/neon/Documents/exjobb/clustering-genomic-signatures/lib/fasta", add_underlines_=False)
    metrics = test_distance_function(d, param_directory, out_dir=None)
    print(metrics)
    for key in metrics_data_vlmc.keys():
      metrics_data_markov_chain[key][datum_index] = metrics[key]
  plt.figure(2)
  for k, v in metrics_data_markov_chain.items():
    x_values = 3 * np.power(4 * np.ones(nbr_data_points), np.array(k_values))
    plt.semilogx(x_values, v, label=k, marker='.')
  plt.grid(True)
  plt.legend()
  plt.show()


def extract_from_data_set_fasta_files(from_start, percentage, fasta_folder="/home/neon/Documents/exjobb/clustering-genomic-signatures/lib/fasta", use_small_data_set=True, db2fasta_folder="../lib"):
  # delete files in "fasta"-directory
  fasta_folder = "/home/neon/Documents/exjobb/clustering-genomic-signatures/lib/fasta"
  os.system("ls -l {}".format(fasta_folder))
  print("Deleting all files in ../lib/fasta/")
  delete_files_in_directory(fasta_folder)
  os.system("ls -l {}".format(fasta_folder))
  
  select = get_select_query(use_small_data_set)
  
  args = ["/usr/bin/perl", "{}/db2fasta.pl".format(db2fasta_folder), "-c", select, "-{}".format(from_start), "-p", str(percentage), "-d", fasta_folder]
  # this subprocess creates fasta files in the "../lib/fasta"-directory
  print("Calling db2fasta")
  popen = subprocess.Popen(args, stdout=subprocess.PIPE)
  popen.wait()
  output = popen.stdout.read()
  print("output from perl")
  print(output)

def get_select_query(use_small_data_set):
  if use_small_data_set:
    print("SMALL DATA SET QUERYR")
    return "\"virus.aid=\'AF147806.2\' OR virus.aid=\'AF147806.2\' OR virus.aid=\'GU980198.1\' OR virus.aid=\'JN133502.1\' OR virus.aid=\'JQ596859.1\' OR virus.aid=\'KF487736.1\' OR virus.aid=\'KJ627438.1\' OR virus.aid=\'NC_001493.2\' OR virus.aid=\'NC_001650.1\' OR virus.aid=\'AB026117.1\' OR virus.aid=\'AC_000005.1\' OR virus.aid=\'AF258784.1\' OR virus.aid=\'JN418926.1\' OR virus.aid=\'KF429754.1\' OR virus.aid=\'NC_000942.1\' OR virus.aid=\'NC_001454.1\' OR virus.aid=\'NC_001734.1\' OR virus.aid=\'NC_002513.1\' OR virus.aid=\'NC_005905.1\' OR virus.aid=\'NC_007767.1\' OR virus.aid=\'NC_008035.3\' OR virus.aid=\'NC_007921.1\' OR virus.aid=\'NC_008293.1\' OR virus.aid=\'NC_008348.1\' OR virus.aid=\'NC_008725.1\' OR virus.aid=\'NC_009011.2\' OR virus.aid=\'KF234407.1\' OR virus.aid=\'NC_001132.2\' OR virus.aid=\'NC_001266.1\' OR virus.aid=\'NC_001611.1\' OR virus.aid=\'NC_001731.1\' OR virus.aid=\'NC_002188.1\' OR virus.aid=\'NC_003389.1\' OR virus.aid=\'NC_003391.1\'\""
  else:
    # I think this is correct for the larger data set?
    print("HELLO?")
    return "\"(LENGTH(seq) > 18000 OR fam=\'Flaviviridae\' OR fam=\'Endornaviridae\' OR fam=\'Hypoviridae\' OR fam=\'Retroviridae\') AND (fam!=\'Spiraviridae\' AND fam!=\'Sphaerolipoviridae\' AND fam!=\'Rudiviridae\' AND fam!=\'Roniviridae\' AND fam!=\'Polydnaviridae\' AND fam!=\'Paramyxoviridae\' AND fam!=\'Myoviridae\' AND fam!=\'Marseilleviridae\' AND fam!=\'Malacoherpesviridae\' AND fam!=\'Hytrosaviridae\' AND fam!=\'Hypoviridae\' AND fam!=\'Globuloviridae\' AND fam!=\'Fuselloviridae\' AND fam!=\'Closteroviridae\' AND fam!=\'Bicaudaviridae\' AND fam!=\'Ascoviridae\')\""


def delete_files_in_directory(directory):
  os.system("rm {}/*".format(directory))


main()
