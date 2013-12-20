import numpy as np
import theano
import theano.tensor as T
import logging
logging.basicConfig(level=logging.DEBUG)

def generate_functions(A, y, gamma):
    tA = T.matrix('A')
    ty = T.col('y')
    tx = T.col('x')
    ttheta = T.col('theta')
    
    tx0 = T.col('x0')
    tx1 = T.col('x1')
    tbetas = T.col('betas')
    
    error = lambda x: (T.dot(tA, x) - ty).norm(2)
    derror = lambda x: T.grad(error(x), x)
    penalty = lambda x: x.norm(1)
    loss = lambda x: error(x) + penalty(x)

    return {
        "select_entering": theano.function([tx],
                                           [T.argmax(abs(derror(tx))), derror(tx)],
                                           givens = {tA: A, ty: y}),
        "qp_optimum": theano.function([tA, tx, ttheta],
                                      T.dot(T.inv(T.dot(tA.T, tA)), T.dot(tA.T, ty) - gamma/2*ttheta),
                                      givens = {ty: y}),
        "find_candidates": theano.function([tbetas, tx0, tx1],
                                           theano.map(lambda b, x0, x1: (1-b)*x0 + b*x1,
                                                      [tbetas], [tx0, tx1])),
        "select_candidate": theano.function([tA, txs],
                                            txs[T.argmax(theano.map(loss, [txs]))],
                                            givens = {ty: y}),
        "optimal_nz": theano.function([tA, tx],
                                      derror(tx) + gamma*tx.sign(),
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
    x = np.zeros(m, 1)
    theta = np.zeros(m, 1)
    active = np.falses(m, 1)

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
        logging.debug("basis %s" % np.array_str(x[active]))

        while True:
            # optimize active variables
            xnew, thetanew = optimize_basis(A[:, active], x[active], theta[active], fs)

            x[active] = xnew
            theta[active] = thetanew
            active[active] = np.logical_not(np.isclose(xnew, 0))

            logging.debug("basis %s" % np.array_str(x[active]))

            # check optimality
            if np.allclose(fs["optimal_nz"](A[:, active], x[active]), 0):
                if not np.all(fs["optimal_z"](A[:, np.logical_not(active)], x[:, np.logical_not(active)]) <= gamma):
                    # let another variable enter
                    break
                else:
                    # optimal
                    return x

def optimize_basis(A, x0, theta, fs):
    x1 = fs["qp_optimum"](A, x0, theta)
    # find zero-crossings
    betas = x0 / (x0 - x1)
    betas = betas[logical_and(0 <= betas, betas < 1)]
    # make sure we investigate x1
    betas[end+1] = 1

    x = fs["select_candidate"](A, fs["find_candidates"](betas, x0, x1))
    theta = np.sign(x)

    return x, theta


l1ls_featuresign(np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]), np.array([0.0, 1.0, 2.0]), 1.0)
