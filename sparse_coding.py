import os
import numpy as np
import logging

from featuresign import l1ls_featuresign
from bases import l2ls_learn_basis_dual

def sparse_coding(X, num_bases, beta, num_iters, iter_callback):
    B = np.random.random((X.shape[0], num_bases)) - 0.5
    B = B / np.sqrt(np.sum(B**2, 0))

    S = np.zeros((num_bases, X.shape[1]))

    for t in xrange(num_iters):
        # shuffle samples
        np.random.shuffle(X.T)

        logging.info("basis %i %s" % (t, B))
        for j in xrange(X.shape[1]):
            logging.info("t %i sample %i %s" % (t, j, X[:, j]))
            S[:, j] = l1ls_featuresign(B, X[:, j], beta, S[:, j])
            logging.info("t %i coding %i %s" % (t, j, S[:, j]))
        S[np.isnan(S)] = 0

        B = l2ls_learn_basis_dual(X, S, 1.0)

        iter_callback(B, S)

    return (B, S)
