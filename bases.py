import os
import numpy as np
import theano
import theano.tensor as T
import theano.sandbox.linalg as la
import scipy.optimize
import logging
logging.basicConfig(level=logging.DEBUG)

def detect_nan(i, node, fn):
    for output in fn.outputs:
        if np.isnan(output[0]).any():
            print '*** NaN detected ***'
            theano.printing.debugprint(node)
            print 'Inputs : %s' % [input[0] for input in fn.inputs]
            print 'Outputs: %s' % [output[0] for output in fn.outputs]
            break

detect_nan_mode = theano.compile.MonitorMode(post_func=detect_nan).excluding('local_elemwise_fusion', 'inplace')

def l2ls_learn_basis_dual(X, S, c):
    tX = T.matrix('X')
    tS = T.matrix('S')
    tc = T.scalar('c')
    tlambdas = T.vector('lambdas')

    tXST = T.dot(tX, tS.T)
    tSSTetc = la.matrix_inverse(T.dot(tS, tS.T) + T.diag(tlambdas))

    objective = -(T.dot(tX, tX.T).trace()
                  - reduce(T.dot, [tXST, tSSTetc, tXST.T]).trace()
                  - tc*tlambdas.sum())

    objective_fn = theano.function([tlambdas],
                                   objective,
                                   givens={tX: X, tS: S, tc: c})
    objective_grad_fn = theano.function([tlambdas],
                                        T.grad(objective, tlambdas),
                                        givens={tX: X, tS: S, tc: c})

    initial_lambdas = 10*np.abs(np.random.random((S.shape[0], 1)))
    output = scipy.optimize.fmin_cg(f=objective_fn,
                                    fprime=objective_grad_fn,
                                    x0=initial_lambdas,
                                    maxiter=100,
                                    full_output=True)
    logging.debug("optimizer stats %s" % (output[1:],))
    logging.debug("optimizer lambdas %s" % output[0])

    lambdas = output[0]
    B = np.dot(np.linalg.inv(np.dot(S, S.T) + np.diag(lambdas)),
               np.dot(S, X.T)).T

    return B
