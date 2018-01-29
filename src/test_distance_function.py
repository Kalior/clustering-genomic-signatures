#! /usr/bin/python
from vlmc import VLMC
from distance import negloglikelihood
import parse_trees_to_json

def test_vlmc_negloglike():
  tree_dir = "../trees"
  parse_trees_to_json.parse_trees(tree_dir)
  vlmcs = VLMC.from_json_dir(tree_dir)
  for vlmc in vlmcs:
    distances = list(map(lambda other: negloglikelihood.distance(vlmc, other), vlmcs))
    closest_vlmc_i = distances.index(min(distances))
    closest_vlmc = vlmcs[closest_vlmc_i]
    print(vlmc.name + "\nis closest to\n" + closest_vlmc.name + "\n" + str(vlmc == closest_vlmc) + "\n\n")


if __name__ == '__main__':
  test_vlmc_negloglike()
