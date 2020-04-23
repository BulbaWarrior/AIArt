#! /bin/python3

import ga
from ga import Chromosome
import numpy as np
import cv2 as cv

population_size = 30
population = [Chromosome(64) for i in range(population_size)]

gen_counter = 0
while True:
    pool = ga.get_mating_pool(population)
   
    pool[0].show()
    offspring = ga.generate_offspring(pool.copy())
    population = pool + offspring
    [chromosome.mutate() for chromosome in population]
    print("finished generation "+ str(gen_counter) + ". Best fitness is: " + str(population[0].fitness()))
    gen_counter += 1
    




