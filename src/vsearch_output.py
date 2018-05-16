#! /usr/bin/python3.6
import os

from vlmc import VLMC
import parse_trees_to_json
from get_signature_metadata import get_metadata_for
from clustering import FromVsearch
from util.draw_clusters import draw_graph


if __name__ == '__main__':
  tree_dir = '../trees_more_192'
  out_directory = '../images/clustering/vsearch'

  try:
    os.stat(out_directory)
  except:
    os.mkdir(out_directory)

  parse_trees_to_json.parse_trees(tree_dir)
  vlmcs = VLMC.from_json_dir(tree_dir)
  metadata = get_metadata_for([vlmc.name for vlmc in vlmcs])

  clustering = FromVsearch(vlmcs, 'vsearch-clusters', metadata)
  clustering_metrics = clustering.cluster()

  pictures = [('Family', 'family'), ('Genus', 'genus'),
              ('Host', 'hosts'), ('Baltimore', 'baltimore')]
  for name, key in pictures:
    draw_graph(clustering_metrics, name, key, 0, out_directory)

  sensitivy, specificity = clustering_metrics.sensitivity_specificity('family')
  percent_family = clustering_metrics.average_percent_same_taxonomy('family')
  percent_genus = clustering_metrics.average_percent_same_taxonomy('genus')
  print("Sensitivity: {}, Specificity: {}, Percent family: {}, Percent genus: {}".format(
      sensitivy, specificity, percent_family, percent_genus))
