#! /bin/python3

import ga
from ga import Chromosome
import numpy as np
import cv2 as cv

population_size = 100
population = [Chromosome() for i in range(population_size)]

gen_counter = 0
while True:
    
    pool = ga.get_mating_pool(population, 30) #top half of the population is considered to be in the mating pool, while the bootom half should die
    population[:30] = pool[:]
    
    population[0].show()
    if (gen_counter % 100 == 0):
        Chromosome.current_image = population[0].get_image()
        population[0].save('generation%d.png'% gen_counter)

    for i in range(30, len(population)):
        population[i] = pool[i % len(pool)].copy()
        population[i].mutate()

    print("finished generation "+ str(gen_counter) + ". Best fitness is: " + str(population[0].fitness()))
    gen_counter += 1
    




