#! /bin/python3

import ga
from ga import Chromosome
import numpy as np
import cv2 as cv

population_size = 40
population = [Chromosome(256) for i in range(population_size)]

gen_counter = 0
while True:
    ga.get_mating_pool(population) #top half of the population is considered to be in the mating pool, while the bootom half should die
    population[population_size//2:] = population[:population_size//2] # copy the top half of the population
   
    #population[0].show()
    if (gen_counter % 100 == 0):
        population[0].save('generation%d.png'% gen_counter)
    ga.generate_offspring(population[:population_size//2]) # the top half becomes the children

    [chromosome.mutate() for chromosome in population[population_size//2:]]
    print("finished generation "+ str(gen_counter) + ". Best fitness is: " + str(population[0].fitness()))
    gen_counter += 1
    




