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

sample_chromosome = np.random.randint(1, 21000, 64, dtype='uint16')
MAX_FITNESS = 20000000

input_hsv = cv.cvtColor(input_img, cv.COLOR_BGR2HSV)

def fitness(chromosome):
    img = generate_image(chromosome)
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    hsv_diff = np.asarray(hsv[:,:,:], dtype=np.int32)
    hsv_diff = hsv_diff - input_hsv
    hsv_diff = np.square(hsv_diff)
    max_fitness = np.full((512, 512), MAX_FITNESS, dtype = np.uint16)
    hue_sum = np.sum(max_fitness - hsv_diff[:,:,0])
    saturation_sum = np.sum(max_fitness - hsv_diff[:,:,1])
    value_sum = np.sum(max_fitness - hsv_diff[:,:,2])
    return (hue_sum*hue_weight + saturation_sum*saturation_weight +
            value_sum*value_weight)


def mutate(chromosome):
    for i in range(len(chromosome)):
        die = np.random.rand()
        if die <= image_mutation_rate:
            chromosome[i] = np.random.randint(1, 21000, dtype=np.uint16)
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
    image = np.zeros((512*upscale_factor,512*upscale_factor,3), np.uint8)
    # deviding the 512x512 image into an 8X8 grid and filling the grid with the images
    # according to the chromosome
    side_size = 512*upscale_factor // 64
    for i in range(side_size**2):
        row = i%side_size
        column = i//side_size
        imname = "data/" + str(chromosome[i]) + ".png"
        image[row*64:row*64 + 64, column*64:column*64 + 64] = cv.imread(imname)
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
        


