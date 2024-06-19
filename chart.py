import os
import numpy as np
import matplotlib.pyplot as plt

import _pickle as cPickle
import json

from config import MAX_NUMBER_OF_ROWS, MAX_NUMBER_OF_COLS

def DBG(*var):
    print('-------------------------')
    print(f"{var=}")
    print('-------------------------')

def printl(list):
    for item in list: print(item)

with open('dict_truncated_resolution.list', 'rb') as file_dict_truncated_resolution:
    dict_truncated_resolution = cPickle.load(file_dict_truncated_resolution)

print('Loaded computations.')

list_found_generators = dict_truncated_resolution['list_found_generators']

matrix_free_resolution_generators = [[0]*MAX_NUMBER_OF_ROWS for k in range(0, MAX_NUMBER_OF_COLS)]
list_matrix_free_resolution_generators = [] 

for i_resolution_module in range(0, len(list_found_generators)):
    for j_relative_degree in range(0, MAX_NUMBER_OF_ROWS):
        for new_generator in list_found_generators[i_resolution_module]:
            if i_resolution_module == 0:
                continue
            if new_generator[0] - i_resolution_module == j_relative_degree:
                matrix_free_resolution_generators[MAX_NUMBER_OF_ROWS-i_resolution_module-1][j_relative_degree] = new_generator
                list_matrix_free_resolution_generators.append(
                    {'x': str(i_resolution_module), 'y': str(j_relative_degree), 'val': str(new_generator)}
                )

printl(matrix_free_resolution_generators)

main_figure = plt.figure(figsize=(40,24))
plt.axis([-1, MAX_NUMBER_OF_COLS, -1, MAX_NUMBER_OF_COLS])

for i_resolution_module in range(0, len(list_found_generators)):
    for j_relative_degree in range(0, MAX_NUMBER_OF_ROWS):
        matrix_entry = matrix_free_resolution_generators[i_resolution_module][j_relative_degree]
        
        if not matrix_entry == 0 or (i_resolution_module == MAX_NUMBER_OF_ROWS-1 and j_relative_degree == 0):
            plt.plot(j_relative_degree, MAX_NUMBER_OF_ROWS-i_resolution_module-1,'bo') 

axis = main_figure.gca()
axis.set_xlabel('t - s (relative degree)')
axis.set_ylabel('s (resolution module index)')

plt.show()

with open("matrix_free_resolution_generators.json", "w") as file_json_matrix_free_resolution_generators:
        json.dump(list_matrix_free_resolution_generators, file_json_matrix_free_resolution_generators)
