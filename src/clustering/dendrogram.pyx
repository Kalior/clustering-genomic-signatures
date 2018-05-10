import scipy
from scipy.cluster.hierarchy import average, dendrogram
import time
import numpy as np
import matplotlib.pyplot as plt
import os

cimport numpy as np

FLOATTYPE = np.float32

from graph_based_clustering cimport GraphBasedClustering
from graph_based_clustering import GraphBasedClustering

cdef class DendrogramClustering(GraphBasedClustering):
  cdef void _cluster(self, num_clusters, distances):
    labels = ["{:>30} {:>30}".format(
        self.metadata[v.name]['genus'],
        self.metadata[v.name]['family'])
        for v in self.vlmcs]
    condensed_distances = scipy.spatial.distance.squareform(self.indexed_distances, checks=False)
    z = average(condensed_distances)
    plt.figure(figsize=(25, 200))
    dendrogram(z, labels=labels, leaf_font_size=18, orientation='right', color_threshold=0.15)

    plt.tight_layout()

    file_name = 'dendrogram.pdf'
    folder = '../images/'
    file_path = os.path.join(folder, file_name)
    plt.savefig(file_path, dpi='figure', format='pdf')
