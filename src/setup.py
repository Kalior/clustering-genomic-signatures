from distutils.core import setup
from Cython.Build import cythonize

files = ['vlmc/vlmc.pyx', 'distance/naive_parameter_sampling.pyx']

setup(
    name='A variable length markov chain model, with accompanying distance functions.',
    ext_modules=cythonize(files),
)
