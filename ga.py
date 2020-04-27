#!/usr/bin/python3
import numpy as np
import cv2 as cv
import sys

upscale_factor=1

hue_weight = 4
saturation_weight = 1
value_weight = 1
color_mutation_rate = 0.1
point_mutation_rate = 0.1
crossover_rate = 0.03
input_img = cv.imread("input2.png")
max_diff = np.asarray(input_img, dtype=np.int32)
max_diff = np.square(max_diff)
max_diff_sum = np.sum(max_diff)

MAX_FITNESS = 20000000


def mutate_vector(vec, amplitude):
    for val in vec:
        die = np.random.randint(0, 1)
        mutation = np.random.randint(0, amplitude)
        if(die == 0):
            val -= mutation
        else:
            val += mutation
        return vec

class Population():
    current_image = np.zeros((512,512,3), dtype=np.uint8)
    def __init__(self, size):
        self.thickness = np.random.randint(2, 20, (size), dtype=np.int16)
        self.size = size
        self.color = np.random.randint(0, 255, (size, 3), dtype=np.int16)
        self.start_point = np.random.randint(0, 511,(size, 2), dtype=np.int16)
        self.end_point = np.random.randint(0, 511, (size, 2), dtype=np.int16)


    def __repr__(self):
        return str(self.color)
            
    def mutate(self, start): # mutate genes starting from start and on
        self.thickness[start:] += np.random.randint(-3, 4, 1, dtype=np.int16)
        self.thickness[self.thickness < 2] = 2
        self.thickness[self.thickness > 20] = 20
        self.color[start:] += np.random.randint(-10, 11, 3, dtype=np.int16)
        # self.color[self.color > 255] = 255
        # self.color[self.color < 0] = 0
        self.start_point[start:] += np.random.randint(-17, 18, 2, dtype=np.int16)
        self.start_point[self.start_point > 511] = 511
        self.start_point[self.start_point < 0] = 0
        self.end_point[start:] += np.random.randint(-17, 18, 2, dtype=np.int16)
        self.end_point[self.end_point > 511] = 511
        self.end_point[self.end_point < 0] = 0
        

    def fitness(self, i):
        img = self.get_image(i)
        diff = np.array(img[:,:,:], dtype=np.int32)
        diff = diff - input_img
        diff = np.square(diff)
        fitness = np.sum(max_diff - diff)/max_diff_sum
        return fitness

    def get_image(self, i):
        img = np.array(self.current_image)
        start_point = tuple(self.start_point[i])
        end_point = tuple(self.end_point[i])
        color = tuple(map(int, self.color[i]))
        thickness = self.thickness[i]
        cv.line(img, start_point, end_point, color, thickness)
        return img

    def show(self, i):
        res = self.get_image(i) 
        cv.imshow('result', res)
        k = cv.waitKey(0)
        if k == ord('x'):         # wait for 'x' key to exit
            sys.exit()
        elif k == ord('s'): # wait for 's' key to save and exit
            cv.imwrite('output/specimen.png', res)
            cv.destroyAllWindows()

    def save(self,i , name):
        cv.imwrite('output/'+name, self.get_image(i))

    def sort(self):
        population_score = [(self.fitness(i), i)
                            for i in range(self.size)]
        population_score.sort(reverse=True)
        color = [self.color[i[1]] for i in population_score]
        start_point = [self.start_point[i[1]] for i in population_score]
        end_point = [self.end_point[i[1]] for i in population_score]
        self.color = np.array(color)
        self.start_point = np.array(start_point)
        self.end_point = np.array(end_point)
                 
    def reproduce(self, pool_size):
        for i in range(self.size):
            self.color[i] = self.color[i % pool_size]
            self.start_point[i] = self.start_point[i % pool_size]
            self.end_point[i] = self.end_point[i % pool_size]


    
def get_mating_pool(population, num):
    population_score = [(population.fitness(), i)
                        for i in range(len(population))]
    # each member of the population is assign a random value from [1, its fitness]
    # the half with highest scores is the mating pool
    population_score.sort(reverse=True)
    mating_pool = [population[i[1]] for i in population_score]
    return mating_pool

def generate_offspring(mating_pool):
    permutation = np.random.permutation(len(mating_pool))
    offspring = []
    for i in range(0, len(mating_pool)-1, 2):
        offspring = mating_pool[permutation[i]].copy().crossover(mating_pool[permutation[i+1]])
    return mating_pool


