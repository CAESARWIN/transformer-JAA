# -*- coding:utf-8 -*-
import numpy as np
import torch


class Graph():
    def __init__(self, strategy='uniform'):
        self.get_adjacency(strategy)

    def __str__(self):
        return self.A

    def get_adjacency(self, strategy):
        A = np.ones((12, 12))
        '''
        A = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                      [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                      [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                      [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                      [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                      [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                      [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                      [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                      [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                      [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                      [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                      [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
        '''
        # Symmetrically normalize adjacency matrix.
        adj = A
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        D_norm = np.diag(d_inv_sqrt)
        L_sym = np.dot(np.dot(D_norm, adj), D_norm)
        self.A = torch.FloatTensor(L_sym)
