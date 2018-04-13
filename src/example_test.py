#! /usr/bin/python3.6
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

label_size = 20
mpl.rcParams['xtick.labelsize'] = label_size
mpl.rcParams['ytick.labelsize'] = label_size
mpl.rcParams['axes.axisbelow'] = True
mpl.rcParams['font.size'] = 24

from vlmc import VLMC
import parse_trees_to_json
from draw_vlmc import save, save_intersection
from test_distance_function import test_distance_function_
from distance import FrobeniusNorm, NegativeLogLikelihood
from train import train


def examples_test(tree_dir, image_dir, gen_tree_dir, gen_image_dir):
  print("Original trees test")
  example_distance(tree_dir, image_dir)

  print("Regenerating trees")
  regenerate_example_vlmcs(tree_dir, gen_tree_dir)

  print("Regenerated trees test")
  example_distance(gen_tree_dir, gen_image_dir)


def regenerate_example_vlmcs(tree_dir, gen_tree_dir):
  vlmcs = VLMC.from_json_dir(tree_dir)

  sequence_length = 100000
  number_of_parameters = 24
  train(vlmcs, sequence_length, gen_tree_dir, number_of_parameters)


def example_distance(tree_dir, image_dir):
  parse_trees_to_json.parse_trees(tree_dir)
  vlmcs = VLMC.from_json_dir(tree_dir)
  d = FrobeniusNorm()
  metadata = {v.name: {n: v.name for n in ['species', 'family', 'genus']} for v in vlmcs}
  test_distance_function_(d, vlmcs, vlmcs, metadata, image_dir)

if __name__ == '__main__':
  tree_dir = '../new_test_trees'
  image_dir = '../images/new_test_trees_intersection'

  gen_tree_dir = '../new_test_re'
  gen_image_dir = '../images/new_test_trees_re_intersection'

  # Make sure all of these directories exists.
  for d in [image_dir, gen_tree_dir, gen_image_dir]:
    try:
      os.stat(d)
    except:
      os.mkdir(d)

  examples_test(tree_dir, image_dir, gen_tree_dir, gen_image_dir)
