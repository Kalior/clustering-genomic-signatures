import numpy as np
import random
import math
import sys
cimport numpy as np
import traceback
import pdb



cdef class GeneticAlgorithm(object):
  cdef list cluster
  cdef object d
  cdef list cluster_complement
  cdef int populationSize
  cdef int nbr_variables
  cdef int numberOfGenerations
  cdef float p_crossover
  cdef float p_mutation
  cdef float p_creep
  cdef float p_tournament
  cdef int tournament_size
  cdef float creep_step
  cdef float max_value
  cdef float min_value
  

  def __init__(self, nbr_variables, cluster, cluster_complement, d):
    self.cluster = cluster
    self.d = d
    self.cluster_complement = cluster_complement
    self.populationSize = 50
    self.nbr_variables = nbr_variables
    self.numberOfGenerations = 5000
    self.p_crossover = self.nbr_variables / 3
    self.p_mutation = 5 / self.nbr_variables
    self.p_creep = 6 * self.p_mutation
    self.creep_step = 1 / 32
    self.tournament_size = 5
    self.p_tournament = 0.75
    self.min_value = 0.0
    self.max_value = 2.0
    

  cpdef np.ndarray execute_genetic_algorithm(self, ):
    maximum_fitness = -1000000
    best_chromosome = np.ndarray([])
    fitness = np.zeros(self.populationSize)
    population = self.initialise_population()
    for gen in np.arange(0, self.numberOfGenerations):
      if gen % 10 == 0:
        print("Generation {} starting".format(gen))
      if gen % 100 == 0:
        print("Best value so far: {}".format(maximum_fitness))
        print(best_chromosome)
      # Evaluation
      for i in np.arange(0, self.populationSize):
        parameters = population[i, :]
        fitness[i] = self.eval_individual(parameters)
        if fitness[i] > maximum_fitness:
          maximum_fitness = fitness[i]
          best_index = i
          best_chromosome = parameters

      new_generation = population.copy()

      # Tournamentselection with Crossover
      for i in np.arange(0, self.populationSize, 2):
        i1 = self.tournament_selection(fitness)
        i2 = self.tournament_selection(fitness)
        paramOne = population[i1, :]
        paramTwo = population[i2, :]
        if random.random() < self.p_crossover:
          # Crossover
          new1, new2 = self.cross(paramOne, paramTwo)
          new_generation[i, :] = new1
          new_generation[i + 1, :] = new2
        else:
          new_generation[i, :] = paramOne
          new_generation[i + 1, :] = paramTwo

      # Mutation
      for i in np.arange(0, self.populationSize):
        original = new_generation[i, :]
        mutated = self.mutate(original)
        new_generation[i, :] = mutated

      # Elitism
      bestIndividual = population[best_index, :]
      new_generation[0, :] = bestIndividual

      population = new_generation
    print("GA finished, best parameters was:")
    print(best_chromosome)
    print(maximum_fitness)
    return best_chromosome


  cdef np.ndarray initialise_population(self):
    return (self.max_value - self.min_value) * np.random.rand(
        self.populationSize, self.nbr_variables) + self.min_value * np.ones([self.populationSize, self.nbr_variables])


  cdef float eval_individual(self, chromosome):
    nbr_internal_vlmcs = len(self.cluster)
    # sum_i^{n-1} i = n(n-1)/2
    nbr_internal_distances = int(nbr_internal_vlmcs * (nbr_internal_vlmcs - 1) / 2)
    vlmcs_internal_distance_computed = set()
    distances = np.zeros(nbr_internal_distances)

    i = 0
    for vlmc1 in self.cluster:
      for vlmc2 in self.cluster:
        if vlmc2 not in vlmcs_internal_distance_computed and vlmc1 != vlmc2:
          self.d.weight_parameters = chromosome
          distances[i] = self.d.distance(vlmc1, vlmc2)
          i += 1
      vlmcs_internal_distance_computed.add(vlmc1)

    maximum_internal_distance = distances.max()

    nbr_outgoing_distances = len(self.cluster) * len(self.cluster_complement)
    interdistances = np.zeros(nbr_outgoing_distances)
    j = 0
    for vlmc in self.cluster:
      for o_vlmc in self.cluster_complement:
        self.d.weight_parameters = chromosome
        interdistances[j] = self.d.distance(vlmc, o_vlmc)
        j += 1

    minimum_distance_to_other = interdistances.min()

    return minimum_distance_to_other - maximum_internal_distance


  # fitness_values is N x 1 vector
  cdef int tournament_selection(self, fitness_values):
    nbr_individuals = fitness_values.size
    indices = list(range(0, nbr_individuals))

    competitor_indecies = random.choices(indices, k=self.tournament_size)
    tournament_pool = np.zeros([self.tournament_size, 2])
    for index, competitor_index in np.ndenumerate(competitor_indecies):
      tournament_pool[index][0] = fitness_values[competitor_index]
      tournament_pool[index][1] = competitor_index

    tournament_pool = tournament_pool[tournament_pool[:, 0].argsort()]
    someone_won = False
    current_competitor_index = self.tournament_size - 1  # list is sorted in reversed
    while not someone_won and current_competitor_index > 0:
      if random.random() < self.p_tournament:
        someone_won = True
      else:
        current_competitor_index -= 1

    winning_index = tournament_pool[current_competitor_index][1]
    return int(winning_index)


  cdef tuple cross(self, chromosome1, chromosome2):
    nbr_genes = chromosome1.size  # gives nbr of columns if only one row

    crossover_index = math.floor(random.random() * nbr_genes)
    new1 = np.zeros([nbr_genes])
    new2 = np.zeros([nbr_genes])
    for i in np.arange(0, nbr_genes):
      if i <= crossover_index:
        new1[i] = chromosome1[i]
        new2[i] = chromosome2[i]
      else:
        new1[i] = chromosome2[i]
        new2[i] = chromosome1[i]

    return new1, new2


  cdef np.ndarray mutate(self, chromosome):
    # mutate but keep values in interval [min_value, max_value]
    mutated = chromosome.copy()
    # pdb.set_trace()
    # print(mutated)
    for i, _ in np.ndenumerate(mutated):
      if random.random() < self.p_mutation:
        mutated[i] = (self.max_value - self.min_value) * random.random()
      elif random.random() < self.p_creep:
        mutated[i] = mutated[i] + (-self.creep_step + 2 * self.creep_step * random.random())
      if mutated[i] < self.min_value:
        mutated[i] = self.min_value
      if mutated[i] > self.max_value:
        mutated[i] = self.max_value
    return mutated


# def foo(exctype, value, tb):
#   print('My Error Information')
#   print('Type:', exctype)
#   print('Value:', value)
#   print('Traceback:', tb)
#   traceback.print_tb(tb)


# def test_torunament_select():
#   some_fitness = np.array(np.arange(10, 20))
#   some_fitness.shape = (10, 1)
#   print(some_fitness[::-1])
#   winnings = np.zeros([10])
#   print(winnings)
#   for i in np.arange(0, 1000):
#     winIndex = tournament_selection(some_fitness[::-1], 0.9, 2)
#     print(winIndex)
#     winnings[winIndex] = winnings[winIndex] + 1
#   print(winnings)


# def test_mutate():
#   original = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
#   for i in np.arange(100):
#     print(mutate(original, 0.1))


# def test_cross():
#   original1 = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
#   original2 = np.array([6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0])
#   for i in np.arange(100):
#     print(cross(original1, original2))


# def test_initialize_population():
#   print(initialise_population(30, 6, 1, 2))


def test(desired_cluster, desired_far_away_from_cluster, nbr_contexts, d):
  # Find parameter a_1, a_2, ..., a_n such that the distances between
  # the elements in S are very small.
  ga = GeneticAlgorithm(
      nbr_variables=nbr_contexts, cluster=desired_cluster, cluster_complement=desired_far_away_from_cluster, d=d)
  best_params = ga.execute_genetic_algorithm()
  return best_params


# sys.excepthook = foo
# # test_torunament_select()
# # test_mutate()
# # test_cross()
# # test_initialize_population()
