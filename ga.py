#!/usr/bin/python3
import numpy as np
import cv2 as cv
import sys

crossover_rate = 0.03
input_img = cv.imread("input3.png")


MAX_FITNESS = 20000000



class Population():
    shape = input_img.shape
    avg_color = np.sum(input_img, axis=(0,1))//(shape[0]*shape[1])
    base_image = np.full_like(input_img, avg_color,  dtype=np.uint8)
    current_image = np.array(base_image)

    fitness_mat = np.array(current_image, dtype=np.int32)
    fitness_mat -= input_img
    fitness_mat = np.square(fitness_mat)
    current_fitness = np.sum(fitness_mat)
    max_diff = np.asarray(input_img, dtype=np.int32)
    max_diff = np.square(max_diff-current_image)
    max_diff_sum = np.sum(max_diff)

    def __init__(self, size):
        self.thickness = np.random.randint(2, 19, (size), dtype = np.int16)
        self.size = size
        self.color = np.random.randint(0, 255, (size, 3), dtype=np.int16)
        self.start_point = np.random.randint(0, 511,(size, 2), dtype=np.int16)
        self.end_point = np.random.randint(0, 511, (size, 2), dtype=np.int16)

    @classmethod
    def accept_line(cls, self):
        cls.current_image = self.get_image(0)
        cls.fitness_mat = np.array(cls.current_image, dtype=np.int32)
        cls.fitness_mat -= input_img
        cls.fitness_mat = np.square(cls.fitness_mat)
        cls.current_fitness = np.sum(cls.fitness_mat)
        self.__init__(self.size)


    def __repr__(self):
        return str(self.color)
            
    def mutate(self, start): # mutate genes starting from start and on

        self.thickness[start:] += np.random.randint(-4, 5, 1, dtype=np.int16)
        self.thickness[self.thickness > 20] = 20
        self.thickness[self.thickness < 2] = 2
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
        line_location = (img != self.current_image)
        
        diff = img[line_location].astype(np.int32) # extract the line from picture
        diff = diff - input_img[line_location] # Subtract previous pixels of the line
        diff = np.square(diff) # find the square of the difference
        current_line_fitness = self.fitness_mat[line_location]
        total_fitness = self.current_fitness - np.sum(current_line_fitness) + np.sum(diff)
        return total_fitness
    

    def get_image(self, i, line=False):
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
        if k == ord('x'): # wait for 'x' key to exit
            sys.exit()
        elif k == ord('s'): # wait for 's' key to save and exit
            cv.imwrite('output/specimen.png', res)
            cv.destroyAllWindows()

    def save(self,i , name):
        cv.imwrite('output4/'+name, self.get_image(i))

    def sort(self):
        population_score = [(self.fitness(i), i)
                            for i in range(self.size)]
        population_score.sort(reverse=False)
        color = [self.color[i[1]] for i in population_score]
        start_point = [self.start_point[i[1]] for i in population_score]
        end_point = [self.end_point[i[1]] for i in population_score]
        self.color = np.array(color)
        self.start_point = np.array(start_point)
        self.end_point = np.array(end_point)
                 
    def reproduce(self, pool_size):
        for i in range(pool_size, self.size):
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


