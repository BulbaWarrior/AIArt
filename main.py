#!/usr/bin/python3
import numpy as np
import cv2 as cv
import sys

upscale_factor=1
input_img = cv.imread("input.png")

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


