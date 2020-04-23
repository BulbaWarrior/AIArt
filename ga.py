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

class Chromosome():
    thickness = 5
    def __init__(self, length):
        self.length = length
        self.color = np.random.randint(0, 255, (length, 3), dtype=np.int16)
        self.start_point = np.random.randint(0, 511, (length, 2), dtype=np.int16)
        self.end_point = np.random.randint(0, 511, (length, 2), dtype=np.int16)

    def __repr__(self):
        return str(self.color)
            
    def mutate(self):
        self.color += np.random.randint(-3, 3, (self.length, 3), dtype=np.int16)
        self.start_point += np.random.randint(-5, 5, (self.length, 2), dtype=np.int16)
        self.end_point = np.random.randint(-5, 5, (self.length, 2), dtype=np.int16)

    def crossover(self, chromosome):
        for i in range(self.length):
            die = np.random.rand()
            if (die <= .5):
                c = np.array(self.color[i,:])
                self.color[i,:] = chromosome.color[i,:]
                chromosome.color[i,:] = c
                c = np.array(self.start_point[i,:])
                self.start_point[i,:] = chromosome.start_point[i,:]
                chromosome.start_point[i,:] = c
                c = np.array(self.end_point[i,:])
                self.end_point[i,:] = chromosome.end_point[i,:]
                chromosome.end_point[i,:] = c

    def fitness(self):
        img = self.get_image()
        diff = np.array(img[:,:,:], dtype=np.int32)
        diff = diff - input_img
        diff = np.square(diff)
        max_diff = np.full_like(diff, 255**2)
        fitness = np.sum(max_diff - diff)/np.sum(max_diff)
        return fitness

    def get_image(self):
        img = np.zeros((512,512,3), dtype=np.uint8)
        for i in range(self.length):
            start_point = tuple(self.start_point[i,:])
            end_point = tuple(self.end_point[i,:])
            color = tuple(map(int, self.color[i,:]))
            cv.line(img, start_point, end_point, color, self.thickness)
        return img

    
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
        


