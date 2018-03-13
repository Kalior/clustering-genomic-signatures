from distutils.core import setup
from Cython.Build import cythonize
import numpy

files = ['vlmc/vlmc.pyx', 'distance/naive_parameter_sampling.pyx', 'distance/negloglikelihood.pyx',
         'distance/stationary_distribution.pyx', 'distance/acgt.pyx', 'distance/frobenius.pyx',
         'distance/estimate.pyx',
         'clustering/graph_based_clustering.pyx', 'clustering/mst_clustering.pyx',
         'clustering/min_inter_cluster_distance.pyx',
         'clustering/k_means.pyx']

setup(
    name='A variable length markov chain model, with accompanying distance functions.',
    ext_modules=cythonize(files, include_path=[numpy.get_include()]),
)
