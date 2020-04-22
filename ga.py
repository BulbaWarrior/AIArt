#!/usr/bin/python3
import numpy as np
import cv2 as cv
import sys

upscale_factor=1

hue_weight = 4
saturation_weight = 1
value_weight = 1
image_mutation_rate = 0.01
crossover_rate = 0.03
input_img = cv.imread("input.png")


MAX_FITNESS = 20000000

input_hsv = cv.cvtColor(input_img, cv.COLOR_BGR2HSV)

def mutate_vector(vec, amplitude):
    for val in vec:
        die = np.random.randint(0, 1)
        mutation = np.random.randint(0, amplitude)
        if(die == 0):
            val -= mutation
        else:
            val += mutation
        return vec

class Gene():
    color = np.zeros(3)
    start_point = np.zeros(2)
    end_point = np.zeros(2)
    thickness = 5
    def __init__(self):
        self.color = np.random.randint(0, 255, 3, dtype=np.uint8)
        self.start_point = np.random.randint(0, 511, 2)
        self.end_point = np.random.randint(0, 511, 2)

    def __repr__(self):
        return str(tuple(self.color))

    def normalize_point(point):
        for coordinate in point:
            if(coordinate > 511):
                coordinate = 511
            if(coordinate < 0):
                coordinate = 0
                
    def mutate(self):
        self.color = mutate_vector(self.color, 3)
        self.start_point = mutate_vector(self.start_point, 5)
        self.end_point = mutate_vector(self.end_point, 5)


def generate_chromosome():
    return [Gene() for i in range(64)]

def fitness(chromosome):
    img = generate_image(chromosome)
    diff = np.asarray(img[:,:,:], dtype=np.int32)
    diff = diff - input_img
    diff = np.square(diff)
    max_diff = np.full_like(diff, 255**2)
    fitness = np.sum(max_diff - diff)
    return fitness
    

def mutate(chromosome):
    for gene in chromosome:
        gene.mutate()
    return chromosome

def crossover(chrom1, chrom2):
    for i in range(len(chrom1)):
        die = np.random.rand()
        if die <= crossover_rate:
            c = chrom1[i]
            chrom1[i] = chrom2[i]
            chrom2[i] = c
    return chrom1, chrom2

def get_mating_pool(population):
    population_score = [(np.random.randint(1, fitness(population[i])), i)
                        for i in range(len(population))]
    # each member of the population is assign a random value from [1, its fitness]
    # the half with highest scores is the mating pool
    population_score.sort(reverse=True)
    mating_pool = [population[i[1]] for i in population_score[:len(population)//2]]
    return mating_pool

def generate_offspring(mating_pool):
    permutation = np.random.permutation(len(mating_pool))
    for i in range(0, len(mating_pool)-1, 2):
        crossover(mating_pool[permutation[i]], mating_pool[permutation[i+1]])
    return mating_pool

def generate_image(chromosome):
    image = np.zeros((512, 512,3), np.uint8)
    for gene in chromosome:
        start_point = tuple(gene.start_point)
        end_point = tuple(gene.end_point)
        color = tuple(map(int, gene.color))
        cv.line(image, start_point, end_point, color, gene.thickness)
    return image

def show_chromosome(chromosome):
    res = generate_image(chromosome) 
    cv.imshow('result', res)
    k = cv.waitKey(0)
    if k == ord('x'):         # wait for 'x' key to exit
        sys.exit()
    elif k == ord('s'): # wait for 's' key to save and exit
        cv.imwrite('output/specimen.png', res)
        cv.destroyAllWindows()
        


