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
import matplotlib as mpl
from matplotlib.lines import Line2D

if __name__ == '__main__':
  label_size = 20 * 3
  mpl.rcParams['xtick.labelsize'] = label_size
  mpl.rcParams['ytick.labelsize'] = label_size
  mpl.rcParams['axes.axisbelow'] = True
  mpl.rcParams['font.size'] = 24 * 3


metrics_used = ["average_procent_of_genus_in_top", "average_procent_of_family_in_top",
                "total_average_distance_to_genus", "total_average_distance_to_family",
                "total_average_distance"]

# k_values is all different "orders" to test for the full order markov chain
# test vlmcs in range(min_param, max_param, step=step_size)


def main(d=FrobeniusNorm(), use_small_data_set=True, min_param=1000,
         max_param=10000, step_size=1000, k_values=[2, 3, 4, 5, 6, 7, 8],
         directory_trained_models="../lib/trainedmodels", min_count=4,
         max_depth=15, from_start='f', percentage=100, image_directory='../images',
         fasta_folder="../lib/fasta"):
  file_of_vlmc_names = create_list_file_of_vlmc_names(
      fasta_folder, from_start, percentage, use_small_data_set)
  parameters = get_default_params(min_count, max_depth)

  # all_parameters_to_test = range(min_param, max_param, step_size)
  all_parameters_to_test = [int(l) for l in np.logspace(1, 6, 10)]
  plot_vlmc_metrics(all_parameters_to_test, parameters, file_of_vlmc_names,
                    fasta_folder, d, image_directory, directory_trained_models)
  plot_mc_metrics(k_values, parameters, file_of_vlmc_names,
                  fasta_folder, d, image_directory, directory_trained_models)


def extract_from_data_set_fasta_files(from_start, percentage,
                                      fasta_folder="../lib/fasta",
                                      use_small_data_set=True,
                                      db2fasta_folder="../lib"):
  # delete files in "fasta"-directory

  delete_files_in_directory(fasta_folder)

  select = get_select_query(use_small_data_set)

  args = ["/usr/bin/perl", "{}/db2fasta.pl".format(db2fasta_folder),
          "-c", select, "-{}".format(from_start), "-p", str(percentage), "-d", fasta_folder]
  # this subprocess creates fasta files in the "../lib/fasta"-directory
  popen = subprocess.Popen(args, stdout=subprocess.PIPE)
  popen.wait()
  output = popen.stdout.read()


def get_select_query(use_small_data_set):
  if use_small_data_set:
    query = "(organism='virus' AND (virus.aid=\'AF147806.2\' OR virus.aid=\'AF147806.2\' OR virus.aid=\'GU980198.1\' OR virus.aid=\'JN133502.1\' OR virus.aid=\'JQ596859.1\' OR virus.aid=\'KF487736.1\' OR virus.aid=\'KJ627438.1\' OR virus.aid=\'NC_001493.2\' OR virus.aid=\'NC_001650.1\' OR virus.aid=\'AB026117.1\' OR virus.aid=\'AC_000005.1\' OR virus.aid=\'AF258784.1\' OR virus.aid=\'JN418926.1\' OR virus.aid=\'KF429754.1\' OR virus.aid=\'NC_000942.1\' OR virus.aid=\'NC_001454.1\' OR virus.aid=\'NC_001734.1\' OR virus.aid=\'NC_002513.1\' OR virus.aid=\'NC_005905.1\' OR virus.aid=\'NC_007767.1\' OR virus.aid=\'NC_008035.3\' OR virus.aid=\'NC_007921.1\' OR virus.aid=\'NC_008293.1\' OR virus.aid=\'NC_008348.1\' OR virus.aid=\'NC_008725.1\' OR virus.aid=\'NC_009011.2\' OR virus.aid=\'KF234407.1\' OR virus.aid=\'NC_001132.2\' OR virus.aid=\'NC_001266.1\' OR virus.aid=\'NC_001611.1\' OR virus.aid=\'NC_001731.1\' OR virus.aid=\'NC_002188.1\' OR virus.aid=\'NC_003389.1\' OR virus.aid=\'NC_003391.1\'))"
    return query
  else:
    # I think this is correct for the larger data set?
    return "(organism='virus' AND (LENGTH(seq) > 18000 OR fam=\'Flaviviridae\' OR fam=\'Endornaviridae\' OR fam=\'Hypoviridae\' OR fam=\'Retroviridae\') AND (fam!=\'Spiraviridae\' AND fam!=\'Sphaerolipoviridae\' AND fam!=\'Rudiviridae\' AND fam!=\'Roniviridae\' AND fam!=\'Polydnaviridae\' AND fam!=\'Paramyxoviridae\' AND fam!=\'Myoviridae\' AND fam!=\'Marseilleviridae\' AND fam!=\'Malacoherpesviridae\' AND fam!=\'Hytrosaviridae\' AND fam!=\'Hypoviridae\' AND fam!=\'Globuloviridae\' AND fam!=\'Fuselloviridae\' AND fam!=\'Closteroviridae\' AND fam!=\'Bicaudaviridae\' AND fam!=\'Ascoviridae\'))"


def delete_files_in_directory(directory):
  os.system("rm {}/*".format(directory))


def create_list_file_of_vlmc_names(fasta_folder, from_start, percentage, use_small_data_set):
  # start by extracting the fasta files into "../lib/fasta"-directory
  # assume db2fasta lies in working directory "."
  file_of_vlmc_names = "/tmp/list.txt"  # had trouble using local file
  tree_files_exists = False
  if not tree_files_exists:
    extract_from_data_set_fasta_files(
        from_start, percentage, use_small_data_set=use_small_data_set)
  os.system("ls -1 {}/*.fa | /bin/sed 's!.*/!!' | /bin/sed 's/\.fa$//' > {}".format(fasta_folder, file_of_vlmc_names))
  return file_of_vlmc_names


def create_data_matrices(nbr_data_points):
  data = {
      "Percent of genus": np.zeros(nbr_data_points),
      "Percent of family": np.zeros(nbr_data_points),
      "Distance to genus": np.zeros(nbr_data_points),
      "Distance to family": np.zeros(nbr_data_points),
      "Average distance": np.zeros(nbr_data_points)
  }
  return data


def get_pretty_name(metric):
  return {
      "average_procent_of_genus_in_top": "Percent of genus",
      "average_procent_of_family_in_top": "Percent of family",
      "total_average_distance_to_genus": "Distance to genus",
      "total_average_distance_to_family": "Distance to family",
      "total_average_distance": "Average distance"
  }.get(metric)


def get_default_params(min_count, max_depth):
  parameters = {
      'use_constant_cutoff': "false",
      'cutoff_value': "3.9075",
      'number_of_parameters': 0,
      'min_count': min_count,
      'max_depth': max_depth,
      # false = Dalevi's way of calculating the nbr of parameters
      'count_free_parameters_individually': "false",
      'generate_full_markov_chain': False,
      'markov_chain_order': 3
  }
  return parameters


def plot_vlmc_metrics(all_parameters_to_test, parameters, file_of_vlmc_names,
                      fasta_folder, d, image_directory, directory_trained_models):
  nbr_data_points = len(all_parameters_to_test)

  metrics_data_vlmc = create_data_matrices(nbr_data_points)

  for datum_index, nbr_param in enumerate(all_parameters_to_test):
    parameters['number_of_parameters'] = nbr_param
    param_directory = directory_trained_models + "/vlmc_{}_parameters".format(nbr_param)
    if not os.path.exists(param_directory):
      # if we haven't already generated these, models, generate them
      os.makedirs(param_directory)
      train_vlmcs(parameters, file_of_vlmc_names, param_directory,
                  input_directory=fasta_folder, add_underlines_=False)
    metrics = test_distance_function(d, param_directory, out_dir=None)
    print(metrics)
    for key in metrics_used:
      metrics_data_vlmc[get_pretty_name(key)][datum_index] = metrics[key]

  param_fig, param_ax = plt.subplots(1, figsize=(30, 20), dpi=80)

  linestyles = ['-', '--', '-.', ':', '--']
  linewidths = [4, 4, 4, 4, 7]
  colors = ['#e53935', '#8E24AA', '#3949AB', '#039BE5', '#00897B']
  labels = []

  for i, (k, v) in enumerate(metrics_data_vlmc.items()):
    labels.append(k)
    param_ax.semilogx(all_parameters_to_test, v, label=k, markersize=0, marker=None,
                      color=colors[i], linewidth=linewidths[i], linestyle=linestyles[i])

  legend_markers = [legend_marker(labels[i], linestyles[i], colors[i], linewidths[i])
                    for i in range(len(metrics_data_vlmc.items()))]

  param_ax.legend(handles=legend_markers, fontsize=30, markerscale=0, loc=2)
  param_ax.set_ylim(0, 1.1)
  param_ax.grid(color='#cccccc', linestyle='--', linewidth=1)

  out_file_vlmc = os.path.join(image_directory, 'vlmc-metrics-increasing-parameters.pdf')
  plt.savefig(out_file_vlmc, bbox_inches='tight', dpi='figure', format='pdf')
  plt.close()


def plot_mc_metrics(k_values, parameters, file_of_vlmc_names,
                    fasta_folder, d, image_directory, directory_trained_models):
  nbr_data_points = len(k_values)
  metrics_data_markov_chain = create_data_matrices(nbr_data_points)

  parameters['generate_full_markov_chain'] = True
  for datum_index, order in enumerate(k_values):
    parameters['markov_chain_order'] = order
    param_directory = directory_trained_models + "/mc_{}_parameters".format(order)
    if not os.path.exists(param_directory):
      # if we haven't already generated these, models, generate them
      os.makedirs(param_directory)
      train_vlmcs(parameters, file_of_vlmc_names, param_directory,
                  input_directory=fasta_folder, add_underlines_=False)
    metrics = test_distance_function(d, param_directory, out_dir=None)
    print(metrics)
    for key in metrics_used:
      metrics_data_markov_chain[get_pretty_name(key)][datum_index] = metrics[key]

  linestyles = ['-', '--', '-.', ':', '--']
  linewidths = [4, 4, 4, 4, 7]
  colors = ['#e53935', '#8E24AA', '#3949AB', '#039BE5', '#00897B']
  labels = []

  markov_fig, markov_ax = plt.subplots(1, figsize=(30, 20), dpi=80)
  for i, (k, v) in enumerate(metrics_data_markov_chain.items()):
    labels.append(k)
    x_values = 3 * np.power(4 * np.ones(nbr_data_points), np.array(k_values))
    markov_ax.semilogx(x_values, v, label=k, markersize=0, marker=None,
                       color=colors[i], linewidth=linewidths[i], linestyle=linestyles[i])

  legend_markers = [legend_marker(labels[i], linestyles[i], colors[i], linewidths[i])
                    for i in range(len(metrics_data_markov_chain.items()))]

  markov_ax.legend(handles=legend_markers, fontsize=30, markerscale=0, loc=2)

  markov_ax.set_ylim(0, 1.1)
  markov_ax.grid(color='#cccccc', linestyle='--', linewidth=1)

  out_file_markov = os.path.join(image_directory, 'markov-metrics-increasing-parameters.pdf')
  plt.savefig(out_file_markov, bbox_inches='tight', dpi='figure', format='pdf')
  plt.close()


def legend_marker(label, linestyle, color, linewidth):
  return Line2D([0], [0], marker=None, linestyle=linestyle, linewidth=linewidth,
                markerfacecolor=color, color=color, label=label)

if __name__ == '__main__':
  k_values = [2, 3, 4, 5, 6, 7, 8]
  min_param = 100
  max_param = 1000
  step_size = 100
  directory_trained_models = '../lib/trainedmodels'
  main(min_param=min_param, max_param=max_param, step_size=step_size,
       k_values=k_values, directory_trained_models=directory_trained_models,
       use_small_data_set=True)
