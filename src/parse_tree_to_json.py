import json
import re

def parse_file(file):
  graph = {}
  with open(file) as f:
    for line in f:
      if line[0:5] == "Node:":
        key, children = parse_line(line)
        graph[key] = children
  return json.dumps(graph)

def parse_line(line):
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

j = parse_file('M3_AB026117.1_10229_34094.tree')
#vlmc = VLMC.from_json(j)
#print(j.negative_log_likelihood("ACCACAGT"))
