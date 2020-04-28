#! /bin/python3

import ga
from ga import Population
import numpy as np
import cv2 as cv
import sys

population_size = 100
pool_size = 30
population = Population(100)

gen_counter = 0
fitness = 0
fail_counter = 0
while True:
    
    population.sort()
    
    # population.show(0)

    if (gen_counter % 30 == 0):
        if(population.fitness(0) < population.current_fitness):
            population.save(0, 'generation%d.png'% gen_counter)
            population.accept_line(population)
            fail_counter = 0
        else:
            fail_counter += 1
            if(fail_counter >= 5):
                print("algorithm finished")
                sys.exit()

    population.reproduce(pool_size)
    population.mutate(pool_size)
    fitness = population.fitness(0)
    print("finished generation "+ str(gen_counter) + ". Best fitness is: " + str(population.fitness(0)))
    gen_counter += 1
    




