import numpy as np
from scipy import sparse
from scipy.sparse import csr_matrix, dok_matrix

class FileReader(object):

    def __init__(self, file_path):
        self.file_path = file_path

    def set_file_path(self, new_path):
        self.file_path = new_path

    def get_file_path(self):
        return self.file_path

    def get_graph_representation(self):
        words_graph = dict()
        dict_new_values = dict()
        index = 0
        with open(self.file_path, 'r') as a_f:
            lines = a_f.readlines()[4:]
            for line in lines:
                line = [int(x) for x in line.split()]
                if line[0] not in dict_new_values.keys():
                    dict_new_values[line[0]] = index
                    index+=1
                    words_graph[line[0]] = list()
                if line[1] not in dict_new_values.keys():
                    dict_new_values[line[1]] = index
                    index+=1
                    words_graph[line[1]] = list()
                words_graph[line[0]].append(line[1])
        return words_graph, dict_new_values

    def get_csr_matrix_representation(self):
        rows = list()
        cols = list()
        dict_new_values = dict()
        index = 0
        with open(self.file_path, 'r') as a_f:
            lines = a_f.readlines()[4:]
            for line in lines:
                line = [int(x) for x in line.split()]
                if line[0] not in dict_new_values.keys():
                    dict_new_values[line[0]] = index
                    index += 1
                if line[1] not in dict_new_values.keys():
                    dict_new_values[line[1]] = index
                    index += 1
                rows.append(dict_new_values[line[1]])
                cols.append(dict_new_values[line[0]])
        csr_matrix =sparse.csr_matrix(([True]*len(rows),(rows,cols)),
                                      shape=(len(dict_new_values),len(dict_new_values)))
        return csr_matrix, dict_new_values
