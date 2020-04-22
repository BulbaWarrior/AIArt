#! /bin/python3

import ga
import numpy as np
import cv2 as cv

population_size = 30
population = [np.random.randint(1, 21000, 64, dtype='uint16') for i in range(population_size)]
ga.MAX_FITNESS = ga.fitness(population[0]) + 10000


gen_counter = 0
while True:
    pool = ga.get_mating_pool(population)
   
    ga.show_chromosome(pool[0])
    offspring = ga.generate_offspring(pool.copy())
    population = pool + offspring
    [ga.mutate(chromosome) for chromosome in population]
    print("finished generation "+ str(gen_counter) + ". Best fitness is: " + str(ga.fitness(population[0])))
    gen_counter += 1
    




