from distutils.core import setup
from Cython.Build import cythonize

files = ['vlmc/vlmc.pyx', 'distance/naive_parameter_sampling.pyx', 'distance/negloglikelihood.pyx',
         'distance/stationary_distribution.pyx', 'distance/acgt.pyx', 'distance/statisticalmetric.pyx']

setup(
    name='A variable length markov chain model, with accompanying distance functions.',
    ext_modules=cythonize(files),
)
