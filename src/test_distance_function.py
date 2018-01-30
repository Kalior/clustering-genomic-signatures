#! /usr/bin/python
from vlmc import VLMC
from distance import NegativeLogLikelihood
import parse_trees_to_json
import argparse
import time

def test_vlmc_negloglike(sequence_length):
  d = NegativeLogLikelihood(sequence_length)
  tree_dir = "../trees"
  parse_trees_to_json.parse_trees(tree_dir)
  vlmcs = VLMC.from_json_dir(tree_dir)
  for vlmc in vlmcs:
    start_time = time.time()
    distances = list(map(lambda other: d.distance(vlmc, other), vlmcs))
    elapsed_time = time.time() - start_time
    closest_vlmc_i = distances.index(min(distances))
    closest_vlmc = vlmcs[closest_vlmc_i]
    print(vlmc.name + "\nis closest to\n" + closest_vlmc.name + "\n" + str(vlmc == closest_vlmc)
          + "\nDistance calculated in " + str(elapsed_time) + "s\n\n")


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Tests the negative log-likelihood distance function.')
  parser.add_argument('seqlen', help='The length of the sequences that are generated to calculate the likelihood.', type=int)
  args = parser.parse_args()
  test_vlmc_negloglike(args.seqlen)
