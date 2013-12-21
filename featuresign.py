import os
import numpy as np
import theano
import theano.tensor as T
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

def show_function(f):
    theano.printing.pydotprint(f, ".show_function.png")
    os.system("feh optimal_nz.png")

def generate_functions(A, y, gamma):
    tA = T.matrix('A')
    ty = T.vector('y')
    tx = T.vector('x')
    ttheta = T.vector('theta')
    
    tx0 = T.vector('x0')
    tx1 = T.vector('x1')
    tbetas = T.vector('betas')
    
    error = lambda x: T.sum((T.dot(tA, x) - ty)**2)
    derror = lambda x: T.grad(error(x), x)
    penalty = lambda x: x.norm(1)
    loss = lambda x: error(x) + penalty(x)

    entering_index = T.argmax(abs(derror(tx)))
    txs, _ = theano.map(lambda b, x0, x1: (1-b)*x0 + b*x1,
                        [tbetas], [tx0, tx1])

    return {
        "select_entering": theano.function([tx],
                                           [entering_index, derror(tx)[entering_index]],
                                           givens = {tA: A, ty: y}),
        "qp_optimum": theano.function([tA, ttheta],
                                      T.dot(T.inv(T.dot(tA.T, tA)), T.dot(tA.T, ty) - gamma/2*ttheta),
                                      givens = {ty: y}),
        "txs": theano.function([tbetas, tx0, tx1], txs),
        "select_candidate": theano.function([tA, tbetas, tx0, tx1],
                                            txs[T.argmin(theano.map(loss, [txs])[0])],
                                            givens = {ty: y}),
        "optimal_nz": theano.function([tA, tx],
                                      derror(tx) + gamma*T.sgn(tx),
                                      givens = {ty: y}),
        "optimal_z": theano.function([tA, tx],
                                     abs(derror(tx)),
                                     givens = {ty: y}),
        }

# TODO use sparse representations where appropriate
def l1ls_featuresign(A, y, gamma):
    # rows are examples
    n, m = A.shape

    fs = generate_functions(A, y, gamma)

    # initialization
    x = np.zeros(m)
    theta = np.zeros(m)
    active = np.zeros(m, dtype=bool)

    while True:
        # select entering variable
        i, l = fs["select_entering"](x)
        if l > gamma:
            theta[i] = -1
            active[i] = True
        elif l < -gamma:
            theta[i] = 1
            active[i] = True
        logging.debug("enter %i, grad %f, sign %i" % (i, l, theta[i]))
        logging.debug("x %s theta %s" % (x, theta))

        while True:
            # optimize active variables
            xnew, thetanew = optimize_basis(A[:, active], x[active], theta[active], fs)

            x[active] = xnew
            theta[active] = thetanew
            active[active] = np.logical_not(np.isclose(xnew, 0))

            logging.debug("x %s theta %s" % (x, theta))

            # check optimality
            optimal_nz = fs["optimal_nz"](A[:, active], x[active])
            logging.debug("optimal_nz %s" % optimal_nz)
            return
            if np.allclose(optimal_nz, 0):
                optimal_z = fs["optimal_z"](A[:, np.logical_not(active)], x[np.logical_not(active)])
                logging.debug("optimal_z %s" % optimal_z)
                if not np.all(optimal_z <= gamma):
                    # let another variable enter
                    break
                else:
                    # optimal
                    return x

def optimize_basis(A, x0, theta, fs):
    x1 = fs["qp_optimum"](A, theta)
    logging.debug("qp_optimum %s" % x1)

    # find zero-crossings
    betas = x0 / (x0 - x1)
    betas = betas[np.logical_and(0 <= betas, betas < 1)]
    # make sure we investigate x1
    betas = np.append(betas, 1)
    logging.debug("betas %s xs %s" % (betas, fs["txs"](betas, x0, x1)))

    x = fs["select_candidate"](A, betas, x0, x1)
    theta = np.sign(x)

    return x, theta

l1ls_featuresign(np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]), np.array([0.0, 1.0, 2.0]), 0.5)
