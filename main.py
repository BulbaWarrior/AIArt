#! /bin/python3

import ga
from ga import Population
import numpy as np
import cv2 as cv

population_size = 100
population = Population(100)

gen_counter = 0
while True:
    
    population.sort()
    
    # population.show(0)
    if (gen_counter % 30 == 0):
        Population.current_image = population.get_image(0)
        population.save(0, 'generation%d.png'% gen_counter)
        population = Population(100)

    population.reproduce(30)
    for i in range(30, population.size):
        population.mutate(i)

    print("finished generation "+ str(gen_counter) + ". Best fitness is: " + str(population.fitness(0)))
    gen_counter += 1
    




