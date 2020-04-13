#!/usr/bin/python3
import numpy as np
import cv2 as cv
import sys

upscale_factor=1

hue_weight = 4
saturation_weight = 1
value_weight = 1

input_img = cv.imread("input.png")
input_hsv = cv.cvtColor(input_img, cv.COLOR_BGR2HSV)

def fitness(chromosome):
    img = generate_image(chromosome)
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    hsv_diff = np.asarray(hsv[:,:,:], dtype=np.int32)
    hsv_diff = hsv_diff - input_hsv
    hsv_diff = np.square(hsv_diff)
    max_fitness = np.full((512, 512), np.iinfo(np.uint16).max, dtype = np.uint16)
    hue_sum = np.sum(max_fitness - hsv_diff[:,:,0])
    saturation_sum = np.sum(max_fitness - hsv_diff[:,:,1])
    value_sum = np.sum(max_fitness - hsv_diff[:,:,2])
    return (hue_sum*hue_weight + saturation_sum*saturation_weight +
            value_sum*value_weight)

    

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

sample_chromosome = np.random.randint(1, 21000, 64, dtype='uint16')
res = generate_image(sample_chromosome)

cv.imshow('result', res)
k = cv.waitKey(0)
cv.destroyAllWindows()


