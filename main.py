#! /bin/python3

import ga
import numpy as np
import cv2 as cv

population_size = 20

population = [np.random.randint(1, 21000, 64, dtype='uint16') for i in range(population_size)]


res = ga.generate_image(sample_chromosome) 

cv.imshow('result', res)
k = cv.waitKey(0)
cv.destroyAllWindows()
