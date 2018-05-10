from distutils.core import setup
from Cython.Build import cythonize
import numpy

files = ['vlmc/vlmc.pyx', 'distance/naive_parameter_sampling.pyx', 'distance/negloglikelihood.pyx',
         'distance/stationary_distribution.pyx', 'distance/acgt.pyx', 'distance/frobenius.pyx',
         'distance/estimate.pyx', 'distance/projection.pyx', 'distance/fixed_length_sequence_kl_divergence.pyx',
         'distance/pstmatching.pyx',
         'clustering/graph_based_clustering.pyx', 'clustering/mst_clustering.pyx',
         'clustering/clustering_metrics.pyx',
         'clustering/average_link_clustering.pyx',
         'clustering/k_means.pyx', 'clustering/util.pyx',
         'clustering/fuzzy_similarity_clustering.pyx',
         'clustering/dendrogram.pyx']

setup(
    name='A variable length markov chain model, with accompanying distance functions.',
    ext_modules=cythonize(files, include_path=[numpy.get_include()]),
)
