#! /bin/python3

import ga
from ga import Population
import numpy as np
import cv2 as cv
import sys
import os.path
from os import makedirs

if (len(sys.argv) == 3):
    input_name = sys.argv[1]
    if (os.path.splitext(input_name)[1] == ''):
        input_name += '.png'

    
    output_folder = sys.argv[2]
    if (input_name[-1] != '/'):
        output_folder += '/'
        
elif (len(sys.argv) == 0):
    input_name = None
    output_folder = None
else:
    print('usage: %s <input_img> <output_folder>' % sys.argv[0])
    sys.exit()

if not os.path.exists(output_folder):
        os.makedirs(output_folder)


    

population_size = 120
pool_size = 30
population = Population(population_size, input_name, output_folder)

gen_counter = 0
fitness = 0
fail_counter = 0
while True:
    
    population.sort()
    
    # population.show(0)

    if (gen_counter % 40 == 0):
        if(population.fitness(0) < population.current_fitness):
            population.save(0, 'generation%d.png'% gen_counter)
            population.accept_line()
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
    




