import os
import networkx as nx
import csv
from itertools import product
import numpy as np

from .clustering_metrics import ClusteringMetrics


class FromVsearch:
  file = ""
  vlmcs = []
  metadata = {}

  def __init__(self, vlmcs, file, metadata):
    self.vlmcs = vlmcs
    self.file = file
    self.metadata = metadata

  def cluster(self):
    clusters = {}
    with open(self.file) as f:
      rd = csv.reader(f, delimiter="\t", quotechar='"')
      for row in rd:
        self._parse_row(row, clusters)

    vlmc_clusters = {k: self._aids_to_vlmcs(aids, self.vlmcs) for k, aids in clusters.items()}
    used_vlmcs = [v for vs in vlmc_clusters.values() for v in vs]
    used_vlmcs_names = [v.name for vs in vlmc_clusters.values() for v in vs]
    used_metadata = {k: v for k, v in self.metadata.items() if k in used_vlmcs_names}

    G = nx.Graph()
    G.add_nodes_from(used_vlmcs)

    self._connect_clusters(vlmc_clusters, G)

    zero_distances = np.ones([len(used_vlmcs), len(used_vlmcs)])
    metrics = ClusteringMetrics(G, 0, zero_distances, used_vlmcs, used_metadata)
    return metrics

  def _parse_row(self, row, clusters):
    if row[0] == 'S':
      self._parse_centroid(row, clusters)
    elif row[0] == 'H':
      self._parse_hit(row, clusters)

  def _parse_centroid(self, row, clusters):
    index = int(row[1])
    aid = self._parse_aid(row[8])
    clusters[index] = [aid]

  def _parse_hit(self, row, clusters):
    index = int(row[1])
    aid = self._parse_aid(row[8])
    clusters[index] += [aid]

  def _parse_aid(self, str):
    split = str.split('|')
    return split[3]

  def _aids_to_vlmcs(self, aids, vlmcs):
    return [self._aid_to_vlmc(aid, vlmcs) for aid in aids]

  def _aid_to_vlmc(self, aid, vlmcs):
    for v in vlmcs:
      if v.name == aid:
        return v

  def _connect_clusters(self, clusters, G):
    for cluster in clusters.values():
      pairs = product(cluster, repeat=2)
      for v1, v2 in pairs:
        G.add_edge(v1, v2)
