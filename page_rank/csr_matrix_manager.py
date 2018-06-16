import numpy as np
from scipy import sparse
class CsrMatrixManager(object):

    def __init__(self, path_to_save, matrix_path, csr_mat=None):
        self.path_to_save = path_to_save
        self.matrix_path = matrix_path
        self.csr_mat = csr_mat

    def load_matrix(self):
        loader = np.load()
        edges = int(loader['edges'])
        nodes = int(loader['nodes'])
        return sparse.csr_matrix(
            (np.bool_(np.ones(edges)), loader['indices'], loader['indptr']),
            shape=(nodes, nodes)
        )

    def save_matrix(self):
        if self.csr_mat:
            np.savez(self.path_to_save,
                     nodes=self.csr_mat.shape[0],
                     edges=self.csr_mat.data.size,
                     indices=self.csr_mat.indices,
                     indptr=self.csr_mat.indptr
                     )
