#! /usr/bin/python

import json
import re
import os
import argparse

def parse_trees(directory):
  for file in [f for f in os.listdir(directory) if f.endswith(".tree")]:
    j = _parse_file(os.path.join(directory, file))
    name, _ = os.path.splitext(file)
    new_file_name = name + '.json'
    with open(os.path.join(directory, new_file_name), 'w') as f:
      f.write(j)


def _parse_file(file):
  graph = {}
  with open(file) as f:
    for line in f:
      if line[0:5] == "Node:":
        key, children = _parse_line(line)
        graph[key] = children
  return json.dumps(graph)

def _parse_line(line):
  numbers = re.findall('-?[0-9]+', line)
  children = {}
  # Assumes we're only working with ACGT (in that order)
  total = int(numbers[5])
  children['A'] = int(numbers[1]) / total
  children['C'] = int(numbers[2]) / total
  children['G'] = int(numbers[3]) / total
  children['T'] = int(numbers[4]) / total

  strings = re.findall('[A-Za-z#]+', line)
  key = strings[1]
  if key == '#':
    key = ''
  return key, children


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Process directory of .tree files into .json.')
  parser.add_argument('dir', help='the directory with .tree files')
  args = parser.parse_args()
  parse_trees(args.dir)

