#! /usr/bin/python3.6

import json
import re
import os
import argparse


def parse_trees(directory, deltas=False):
  for file in [f for f in os.listdir(directory) if f.endswith(".tree")]:
    j = _parse_file(os.path.join(directory, file), deltas)
    name, _ = os.path.splitext(file)
    new_file_name = name + '.json'
    with open(os.path.join(directory, new_file_name), 'w') as f:
      f.write(j)


def _parse_file(file, deltas):
  tree = {}
  stationary_distribution = {}
  with open(file) as f:
    total_occurences = -1
    for i, line in enumerate([line for line in f if line[0:5] == "Node:"]):
      key, children, occurences = _parse_line(line, deltas)
      tree[key] = children
      if i == 0:
        total_occurences = occurences
      
      stationary_distribution[key] = occurences / (total_occurences - max(len(key) - 1, 0))

  vlmc = {"tree": tree, "stationary_distribution": stationary_distribution}
  return json.dumps(vlmc)


def _parse_line(line, deltas):
  numbers = re.findall('-?[0-9]+\.?[0-9]*', line)
  children = {}
  # Assumes we're only working with ACGT (in that order)
  total = sum([int(numbers[i]) for i in [6, 7, 8, 9]])
  if deltas:
    children['A'] = float(numbers[14])
    children['C'] = float(numbers[15])
    children['G'] = float(numbers[16])
    children['T'] = float(numbers[17])
  else:
    children['A'] = int(numbers[6]) / total
    children['C'] = int(numbers[7]) / total
    children['G'] = int(numbers[8]) / total
    children['T'] = int(numbers[9]) / total

  strings = re.findall('[A-Za-z#]+', line)
  key = strings[1]
  if key == '#':
    key = ''
  return key, children, total


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Process directory of .tree files into .json.')
  parser.add_argument('--dir', help='the directory with .tree files')
  args = parser.parse_args()
  parse_trees(args.dir, deltas=True)
