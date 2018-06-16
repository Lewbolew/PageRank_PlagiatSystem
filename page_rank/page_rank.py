import numpy as np

class PageRank(object):
    """
    Represents class of the page rank algorithm.
    """
    def __init__(self, num_iterations, comp_sparse_mat=None,words_dict=None):
        self.words_dict = words_dict
        self.comp_sparse_mat = comp_sparse_mat
        self.B = 0.85
        self.num_iterations = num_iterations

    def set_sparse_mat(self, sparse_mat):
        self.comp_sparse_mat = sparse_mat

    def set_words_dict(self, graph):
        self.words_dict = graph

    def run_dict_version(self):
        """
        Run page rank algorithm in graph data representation

        Args:
            None
        Return:
            - ranks(dict): dictionary with web_page-rank key-value pairs.
        """
        d = 0.8
        num_loops = 2
        ranks = {}
        num_pages = len(self.words_dict)
        for page in self.words_dict:
            ranks[page] = 1.0 / num_pages
        for i in range(0, num_loops):
            new_ranks = {}
            for page in self.words_dict:
                new_rank = (1 - d) / num_pages
                for node in self.words_dict:
                    if page in self.words_dict[node]:
                        new_rank = new_rank + d * ranks[node] / len(self.words_dict[node])
                new_ranks[page] = new_rank
            ranks = new_ranks
        return ranks

    def run_sparse_version(self):
        """
        Runs page rank algorithm with compressed sparse matrix data representation.

        Args:
            None
        Returns:
            - ranks(numpy array): array of ranks of the sites.
            - time(int): number of iterations of the Power Method
        """
        N, _ = self.comp_sparse_mat.shape
        deg_out_beta = self.comp_sparse_mat.sum(axis=0).T / self.B  # vector
        ranks = np.ones((N, 1)) / N
        time = 0
        for i in range(self.num_iterations):
            time += 1
            with np.errstate(divide='ignore'):  # Ignore division by 0 on ranks/deg_out_beta
                new_ranks = self.comp_sparse_mat.dot((ranks / deg_out_beta))  # vector
            new_ranks += (1 - new_ranks.sum()) / N
            ranks = new_ranks
        return (ranks, time)


