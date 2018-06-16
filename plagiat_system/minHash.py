import numpy as np
from sklearn.metrics import jaccard_similarity_score


class MinHash(object):
    """
    Computes plagiarism distances between files and write results to the file
    """
    def __init__(self, corpus_handler, num_permutations, save_to):
        self.corpus_handler = corpus_handler
        self.input_matrix = self.corpus_handler.get_input_matrix()
        self.num_permutations = num_permutations
        self.path_to_save = save_to

    def set_new_corpus_handler(self, new_corpus_handler):
        self.corpus_handler = new_corpus_handler
        self.input_matrix = self.corpus_handler.get_input_matrix()

    def set_num_permutations(self, new_num_permutations):
        self.num_permutations = new_num_permutations

    def min_hash(self):
        """
        Minhash algorithm
        :return:
            - signature matrix
        """
        signature_matrix = np.zeros((self.num_permutations,
                                     self.input_matrix.shape[1]),
                                    dtype='uint')

        orig_range = np.arange(len(self.corpus_handler.all_corpus_shingles))
        signature_matrix_row_index = 0

        for i in range(self.num_permutations):
            new_permutation = np.random.permutation(orig_range)
            num_row_in_permutation = 0
            filling_value = 1
            while len(signature_matrix[signature_matrix_row_index][
                                  signature_matrix[signature_matrix_row_index] == 0]) != 0:
                next_row_index = np.where(new_permutation == num_row_in_permutation)[0][0]
                zero_positions_row_signature_matrix = np.argwhere(
                    signature_matrix[signature_matrix_row_index] == 0).flatten()
                ones_in_original_matrix = np.argwhere(self.input_matrix[next_row_index] == 1).flatten()
                places_to_put_new_values = np.intersect1d(
                    ones_in_original_matrix,
                    zero_positions_row_signature_matrix
                )
                np.put(signature_matrix[signature_matrix_row_index],
                       places_to_put_new_values,
                       filling_value
                       )
                if (len(places_to_put_new_values) != 0):
                    filling_value += 1
                num_row_in_permutation += 1
            signature_matrix_row_index += 1
        self.signature_matrix = signature_matrix
        return signature_matrix

    def get_plagiat_pairs(self):
        """
        Computes jaccard distances between each files and find most close pairs
        :return:
            - Dictionary most closest pairs
        """
        plagiat_pairs = dict()
        for i in range(len(self.corpus_handler.file_names)):
            distances = [jaccard_similarity_score(self.signature_matrix[:, i],
                                                  self.signature_matrix[:, x])
                         for x in range(self.input_matrix.shape[1])]
            index_file_itself = distances.index(max(distances))
            distances[index_file_itself] = -1
            plagiat_file = max(distances)
            index_plag_file = distances.index(plagiat_file)
            plagiat_pairs[self.corpus_handler.file_names[i]] = self.corpus_handler.file_names[index_plag_file]
        self.plagiat_pairs = plagiat_pairs
        return plagiat_pairs

    def to_txt(self):
        """
        Write result to the file
        :return:
            - None
        """
        info_to_write = list(zip(self.plagiat_pairs.keys(), self.plagiat_pairs.values()))
        with open(self.path_to_save + 'plagiat_pairs.txt', 'a') as the_file:
            for pair in info_to_write:
                the_file.write(pair[0] + ' - ' + pair[1] + '\n')