#! /usr/bin/python3.6

import parse_trees_to_json
from get_signature_metadata import get_metadata_for
from vlmc import VLMC

import json
import numpy as np
from collections import Counter


def print_relations(vlmcs, metadata):
  for i, vlmc in enumerate(vlmcs[8:9]):
    print(i, metadata[vlmc.name]['species'])
    same_family = [other for other in vlmcs if metadata[
        other.name]['family'] == metadata[vlmc.name]['family'] and other.name != vlmc.name]
    same_genus = [other for other in vlmcs if metadata[
        other.name]['genus'] == metadata[vlmc.name]['genus'] and other.name != vlmc.name]

    probs = from_tree_to_list(vlmc)
    print("Same family:")
    print_stationary_differences(metadata, same_family, vlmc)
    print("Same genus")
    print_stationary_differences(metadata, same_genus, vlmc)


def print_stationary_differences(metadata, same_taxonomy, vlmc):
  vlmc_distribution = vlmc.estimated_context_distribution(100000)
  for other in same_taxonomy:
    print(metadata[other.name]['species'], other.name)
    other_distribution = other.estimated_context_distribution(100000)
    for i, ctx in enumerate(other_distribution.keys()):
      if ctx in vlmc_distribution:
        s = abs(other_distribution[ctx] - vlmc_distribution[ctx])
        if s < 0.001:
          print("{:>8} {:5.5f}".format(ctx, s))

    print("\n")


def print_probability_differences(metadata, same_taxonomy, vlmc):
  for other in same_taxonomy:
    print(metadata[other.name]['species'])
    for i, ctx in enumerate(other.tree.keys()):
      if ctx in vlmc.tree:
        for (c, p) in other.tree[ctx].items():
          s = abs(p - vlmc.tree[ctx][c])
          if s < 0.04:
            print("{:>8} {} {:5.5f}".format(ctx, c, s))

    print("\n")


def from_tree_to_list(vlmc):
  return [(context, c, p) for context in vlmc.tree.keys() for (c, p) in vlmc.tree[context].items()]


def number_in_ranks(metadata):
  ranks = ['organism', 'order', 'family', 'subfamily', 'genus']
  for rank in ranks:
    number_in_rank(metadata, rank)

  length_family = ([(v['sequence_length'], v['family']) for k, v in metadata.items()])
  lengths = np.array([v['sequence_length'] for v in metadata.values()])
  print("Sequence length, min: {}, max: {}, mean: {}, median: {}".format(
      lengths.min(), lengths.max(), lengths.mean(), np.median(lengths)))
  length_family = sorted(length_family, key=lambda k: k[0])
  print(np.array_split(length_family, 10))


def number_in_rank(metadata, key):
  rank = Counter([v[key] for v in metadata.values()])
  print("{}: size {}\n{}".format(key, len(rank.keys()), rank))


if __name__ == '__main__':
  tree_dir = '../trees_mixed_192'
  parse_trees_to_json.parse_trees(tree_dir)
  vlmcs = VLMC.from_json_dir(tree_dir)
  metadata = get_metadata_for([vlmc.name for vlmc in vlmcs])
  vlmcs = [v for v in vlmcs if metadata[v.name]['organism'] == 'virus']
  metadata = {k: v for k, v in metadata.items() if v['organism'] == 'virus'}
  number_in_ranks(metadata)
