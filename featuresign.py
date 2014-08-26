import os
import numpy as np
import theano
import theano.tensor as T
import theano.sandbox.linalg as tl
import logging

def detect_nan(i, node, fn):
    for output in fn.outputs:
        if np.isnan(output[0]).any():
            print '*** NaN detected ***'
            theano.printing.debugprint(node)
            print 'Inputs : %s' % [input[0] for input in fn.inputs]
            print 'Outputs: %s' % [output[0] for output in fn.outputs]
            break

detect_nan_mode = theano.compile.MonitorMode(post_func=detect_nan).excluding('local_elemwise_fusion', 'inplace')

def show_function(f):
    theano.printing.pydotprint(f, ".show_function.png")
    os.system("feh optimal_nz.png")

class Thing(object):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

def generate_functions(A, y, gamma):
    full = Thing(A=theano.shared(A),
                 x=T.vector('x'),
                 theta=T.vector('theta'))

    # tpart should be a vector of integer indices (e.g. active set indices)
    tpart = T.lvector('part')

    partial = Thing(A=full.A[:, tpart],
                    x=full.x[tpart],
                    theta=full.theta[tpart])

    error = lambda x: ((T.dot(partial.A, x) - y)**2).sum()
    derror = lambda x: T.grad(error(x), x)
    penalty = lambda x: gamma*abs(x).sum()
    loss = lambda x: error(x) + penalty(x)
    dloss = lambda x: T.grad(loss(x), x)
    loss_if_consistent = lambda x: error(x) + gamma*T.dot(partial.theta, x)

    def generate_select_entering():
        entering_index = T.argmax(abs(derror(partial.x)))
        return theano.function([tpart, full.x],
                               [tpart[entering_index],
                                derror(partial.x)[entering_index]])

    def generate_optimize_basis():
        # original solution
        tx0 = partial.x
        # optimized solution
        tx1 = T.dot(tl.matrix_inverse(T.dot(partial.A.T, partial.A)),
                    T.dot(partial.A.T, y) - gamma/2*partial.theta)

        # investigate zero crossings between tx0 and tx1
        tbetas = tx0 / (tx0 - tx1)
        # investigate tx1
        tbetas = T.concatenate([tbetas, [1.0]])
        # only between tx0 and inclusively tx1
        tbetas = tbetas[(T.lt(0, tbetas) * T.le(tbetas, 1)).nonzero()]

        txbs, _ = theano.map(lambda b: (1-b)*tx0 + b*tx1, [tbetas])
        tlosses, _ = theano.map(loss, [txbs])
        # select the optimum
        txb = txbs[T.argmin(tlosses)]

        return theano.function([tpart, full.x, full.theta],
                               [T.set_subtensor(partial.x,     txb),
                                T.set_subtensor(partial.theta, T.sgn(txb))])

    return {
        "loss": theano.function([tpart, full.x],
                                loss(partial.x)),
        "loss_if_consistent": theano.function([tpart, full.x, full.theta],
                                              loss_if_consistent(partial.x)),
        "select_entering": generate_select_entering(),
        "optimize_basis": generate_optimize_basis(),
        "optimal_nz": theano.function([tpart, full.x], dloss(partial.x)),
        }

# TODO use sparse representations where appropriate
def l1ls_featuresign(A, y, gamma, x=None):
    # rows are examples
    n, m = A.shape

    fs = generate_functions(A, y, gamma)

    # initialization
    if x is None:
        x = np.zeros(m)
    theta = np.sign(x)
    active = np.abs(theta, dtype=bool)
    basis_optimal = False

    while True:
        zero_mask = x == 0
        if np.any(zero_mask):
            # select entering variable
            part = np.nonzero(zero_mask)[0]
            i, l = fs["select_entering"](part, x)

            if abs(l) > gamma:
                theta[i] = -np.sign(l)
                active[i] = True
                basis_optimal = False
                logging.debug("enter %i, grad %f, gamma %f, sign %i" % (i, l, gamma, theta[i]))
            elif basis_optimal:
                logging.debug("optimal")
                break
            elif not np.any(active):
                logging.debug("empty basis and no entering variable")
                break
            else:
                logging.debug("no entering variable")
                break
        elif basis_optimal:
            logging.debug("optimal")
            break

        logging.debug("x %s theta %s" % (x, theta))

        while not basis_optimal:
            # optimize active variables
            part = np.nonzero(active)[0]
            x, theta = fs["optimize_basis"](part, x, theta)

            active[active] = np.logical_not(np.isclose(x[active], 0))
            part = np.nonzero(active)[0]

            logging.debug("x %s theta %s" % (x[active], theta[active]))

            # check optimality
            optimal_nz = fs["optimal_nz"](part, x)
            logging.debug("optimal_nz %s" % optimal_nz)
            if np.allclose(optimal_nz, 0):
                # maybe let another variable enter
                basis_optimal = True

    logging.debug("final x %s" % x)
    return x
