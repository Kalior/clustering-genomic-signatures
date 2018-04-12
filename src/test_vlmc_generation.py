#! /usr/bin/python3.6
from vlmc import VLMC
import parse_trees_to_json
from draw_vlmc import save, save_intersection
from test_distance_function import calculate_distances
from distance import FrobeniusNorm
from train import train

import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import multiprocessing

label_size = 20
mpl.rcParams['xtick.labelsize'] = label_size
mpl.rcParams['ytick.labelsize'] = label_size
mpl.rcParams['axes.axisbelow'] = True
mpl.rcParams['font.size'] = 24


def calculate_distances_for_lengths(vlmcs, lengths, out_directory, image_directory, pool):
  d = FrobeniusNorm()

  results = [pool.apply_async(pooled_distance_calculation,
                              (length, vlmcs, out_directory, image_directory, d))
             for length in lengths]

  distances = np.stack([res.get() for res in results])

  return distances


def pooled_distance_calculation(length, vlmcs, out_directory, image_directory, d):
  thread_out_directory = os.path.join(out_directory, str(length))
  thread_image_directory = os.path.join(image_directory, str(length))
  for dir_ in [thread_out_directory, thread_image_directory]:
    try:
      os.stat(dir_)
    except:
      os.mkdir(dir_)
  distances = distance_for_length(length, vlmcs, thread_out_directory, thread_image_directory, d)
  return distances


def distance_for_length(length, vlmcs, out_directory, image_directory, d):
  repetitions = 5
  distances = np.zeros(len(vlmcs))
  for _ in range(repetitions):
    train(vlmcs, length, out_directory)

    parse_trees_to_json.parse_trees(out_directory)
    new_vlmcs = VLMC.from_json_dir(out_directory)

    pairs = pair_vlmcs(vlmcs, new_vlmcs)

    # plot_vlmcs(pairs, image_directory)
    rep_distance = distance_calculation(pairs, d)
    distances += rep_distance / repetitions

  print(length + " done")
  return distances


def distance_calculation(pairs, d):
  distances = np.array([d.distance(*p) for p in pairs])
  return distances


def pair_vlmcs(vlmcs, new_vlmcs):
  pairs = [(v1, v2) for v1 in vlmcs for v2 in new_vlmcs if v1.name == v2.name[2:]]
  return pairs


def plot_vlmcs(pairs, image_directory):
  try:
    os.stat(image_directory)
  except:
    os.mkdir(image_directory)

  metadata = {v.name: {'species': v.name} for p in pairs for v in [*p]}
  for p in pairs:
    save_intersection([*p], metadata, image_directory)


def plot_results(vlmcs, results, lengths, image_directory):
  fig, ax = plt.subplots(1, figsize=(50, 30), dpi=80)

  ax.grid(color='#cccccc', linestyle='--', linewidth=1)
  ax.set_xticklabels(lengths)
  plt.xticks(np.arange(len(lengths)))
  ax.set_title('Regeneration distance')

  handles = ax.plot(results, markersize=5, marker='o')

  labels = [v.name for v in vlmcs]
  plt.legend(handles=handles, labels=labels, bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)

  out_file = os.path.join(image_directory, 'distance-regeneration.pdf')
  plt.savefig(out_file, dpi='figure', format='pdf')


def test():
  out_directory = "../test_128"
  in_directory = "../test_trees_128"
  image_directory = "../images/128"

  with multiprocessing.Pool(processes=4) as pool:
    parse_trees_to_json.parse_trees(in_directory)
    vlmcs = VLMC.from_json_dir(in_directory)

    lengths = [int(l) for l in np.logspace(2, 6, 10)]
    print(lengths)
    distances = calculate_distances_for_lengths(
        vlmcs, lengths, out_directory, image_directory, pool)

    plot_results(vlmcs, distances, lengths, image_directory)

if __name__ == "__main__":
  test()
