import os
import numpy as np
import logging
logging.basicConfig(level=logging.DEBUG)

from featuresign import l1ls_featuresign
from bases import l2ls_learn_basis_dual

def sparse_coding(X, num_bases, beta, num_iters, iter_callback):
    B = np.random((X.shape[0], num_bases)) - 0.5
    B = B / np.sqrt(np.sum(B**2, 0))

    S = np.zeros((num_bases, X.shape[1]))

    for t in xrange(num_iters):
        # shuffle samples
        np.random.shuffle(X.T)

        for j, x in enumerate(X.columns):
            S[:, j] = l1ls_featuresign(B, x, beta, S[:, j])
        S[np.isnan(S)] = 0

        B = l2ls_learn_basis_dual(X, S, 1)

        iter_callback(B, S)
