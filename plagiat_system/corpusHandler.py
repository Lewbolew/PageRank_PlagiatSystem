from scipy.sparse import csr_matrix
import numpy as np
import re
import os

class CorpusHandler(object):
    """
    Class for getting shingles, and input matrix and compressed matrix from the file.
    """

    def __init__(self, corpus_location):
        self.corpus_location = corpus_location
        self.file_names = os.listdir(corpus_location)
        self.all_corpus_shingles = self.get_unique_shingles_from_corpus()

    def get_unique_shingles_from_corpus(self):
        """
        Get all unique shingles from corpus.

        Return:
            - dict: all unique shingles
        """
        unique_shingles = dict()
        shingle_index = 0
        for file in self.file_names:
            file_shingles = self._get_unique_shingles_from_file(file)
            for shingle in file_shingles:
                if shingle not in unique_shingles.keys():
                    unique_shingles[shingle] = shingle_index
                    shingle_index += 1
        return unique_shingles

    def _get_unique_shingles_from_file(self, file_name):
        """
        Get all unique shingles from one text file

        Return:
            - list: all unique shingles
        """
        unique_elements = list()
        with open(self.corpus_location + file_name, encoding='latin-1') as a_f:
            lines = a_f.readlines()
            filtered_lines = ' '.join([re.sub(r'[^\w\s]', '', x).strip().lower()
                                       for x in lines
                                       if len(re.sub(r'[^\w\s]', '', x).strip()) != 0])

            for i in range(len(filtered_lines.split()) - 3):
                cur_shingle = filtered_lines.split()[i] + ' ' + \
                              filtered_lines.split()[i + 1] + ' ' + \
                              filtered_lines.split()[i + 2]

                if cur_shingle not in unique_elements:
                    unique_elements.append(cur_shingle)
        return unique_elements

    def get_input_matrix(self):
        """
        Return:
            - np.array: with corpus representation
        """
        input_matrix = np.zeros((len(self.all_corpus_shingles), len(self.file_names)), dtype='int')

        for i in range(len(self.file_names)):
            curr_file = lol = self._get_unique_shingles_from_file(self.file_names[i])
            indexes_of_shingles = [self.all_corpus_shingles[x] for x in lol]
            np.put(input_matrix[:, i], indexes_of_shingles, 1)
        return input_matrix

    def get_compressed_input_matrix(self):
        """
        Return:
            - Compressed input matrix
        """
        return csr_matrix(self.get_input_matrix())