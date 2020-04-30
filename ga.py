#!/usr/bin/python3
import numpy as np
import cv2 as cv
import sys


class Population():
    
    
    img_buffer = np.zeros((512, 512, 3), dtype=np.uint8)

    def __init__(self, size, input_name='input.png', output_folder='output_folder'):
        self.size = size
        self.input_img = cv.imread(input_name)
        shape = self.input_img.shape
        avg_color = np.sum(self.input_img, axis=(0,1))//(shape[0]*shape[1])
        self.base_image = np.full_like(self.input_img, avg_color,  dtype=np.uint8)
        self.current_image = np.array(self.base_image)
        
        fitness_mat = np.array(self.current_image, dtype=np.int32)
        fitness_mat -= self.input_img
        self.fitness_mat = np.square(fitness_mat)
        self.current_fitness = np.sum(fitness_mat)
        max_diff = np.asarray(self.input_img, dtype=np.int32)
        self.max_diff = np.square(max_diff-self.current_image)
        self.max_diff_sum = np.sum(self.max_diff)
        self.output_folder = output_folder
        self.input_img = cv.imread(input_name)

        self.mutate_color = np.random.randint(0, 2, (size), dtype=np.bool)
        self.thickness = np.random.randint(1, 11, (size), dtype=np.int16)
        self.color = np.random.randint(0, 256, (size, 3), dtype=np.int16)
        self.start_point = np.random.randint(0, 512,(size, 2), dtype=np.int16)
        self.end_point = np.random.randint(0, 512, (size, 2), dtype=np.int16)

    def reroll_population(self):
        self.mutate_color = np.random.randint(0, 2, (self.size), dtype=np.bool)
        self.thickness = np.random.randint(1, 11, (self.size), dtype=np.int16)
        self.color = np.random.randint(0, 256, (self.size, 3), dtype=np.int16)
        self.start_point = np.random.randint(0, 512,(self.size, 2), dtype=np.int16)
        self.end_point = np.random.randint(0, 512, (self.size, 2), dtype=np.int16)


    def accept_line(self):
        np.copyto(self.current_image, self.get_image(0))
        fitness_mat = np.array(self.current_image, dtype=np.int32)
        fitness_mat -= self.input_img
        self.fitness_mat = np.square(fitness_mat)
        self.current_fitness = np.sum(self.fitness_mat)
        self.reroll_population()


    def __repr__(self):
        return str(self.color)
            
    def mutate(self, start): # mutate genes starting from start and on
        
        
        self.thickness[start:] += np.random.randint(-7, 8,
                                                    self.thickness[start:].shape, dtype=np.int16)
        self.thickness[self.thickness > 50] = 50
        self.thickness[self.thickness < 1] = 1
        mutate_color = self.mutate_color[start:]
        self.color[start:][mutate_color] += np.random.randint(-50, 51,
                                                              (np.sum(mutate_color), 3),
                                                              dtype=np.int16)
        
        # self.color[self.color > 255] = 255
        # self.color[self.color < 0] = 0
        mutate_position = np.logical_not(self.mutate_color[start:])
        mutate_position_sum = np.sum(mutate_position)
        self.start_point[start:][mutate_position] += np.random.randint(-75, 76,
                                                                       (mutate_position_sum, 2),
                                                                       dtype=np.int16)
        self.start_point[self.start_point > 511] = 511
        self.start_point[self.start_point < 0] = 0
        
        self.end_point[start:][mutate_position] += np.random.randint(-75, 76,
                                                                     (mutate_position_sum, 2),
                                                                     dtype=np.int16)
        self.end_point[self.end_point > 511] = 511
        self.end_point[self.end_point < 0] = 0
        
        self.mutate_color = np.random.randint(0, 2, self.size, dtype=np.bool)
        
        
    def fitness(self, i):
        img = self.get_image(i)
        line_location = (img != self.current_image)
        input_img = self.input_img
        diff = img[line_location].astype(np.int32) # extract the line from picture
        diff = diff - input_img[line_location] # Subtract previous pixels of the line
        diff = np.square(diff) # find the square of the difference
        current_line_fitness = self.fitness_mat[line_location]
        total_fitness = self.current_fitness - np.sum(current_line_fitness) + np.sum(diff)
        return total_fitness
    

    def get_image(self, i):
        np.copyto(self.img_buffer, self.current_image)
        start_point = tuple(self.start_point[i])
        end_point = tuple(self.end_point[i])
        color = tuple(map(int, self.color[i]))
        thickness = self.thickness[i]
        cv.line(self.img_buffer, start_point, end_point, color, thickness)
        return self.img_buffer

    def show(self, i):
        res = self.get_image(i) 
        cv.imshow('result', res)
        k = cv.waitKey(0)
        if k == ord('x'): # wait for 'x' key to exit
            sys.exit()
        elif k == ord('s'): # wait for 's' key to save and exit
            cv.imwrite(self.output_folder + 'specimen.png', res)
            cv.destroyAllWindows()

    def save(self,i , name):
        cv.imwrite(self.output_folder + name, self.get_image(i))

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


