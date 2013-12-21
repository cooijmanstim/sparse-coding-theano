import numpy as np
import theano
import theano.tensor as T
import logging
logging.basicConfig(level=logging.DEBUG)

def generate_functions(A, y, gamma):
    tA = T.matrix('A')
    ty = T.vector('y')
    tx = T.vector('x')
    ttheta = T.vector('theta')
    
    tx0 = T.vector('x0')
    tx1 = T.vector('x1')
    tbetas = T.vector('betas')
    
    error = lambda x: (T.dot(tA, x) - ty).norm(2)
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
        logging.debug("x %s" % x)

        while True:
            # optimize active variables
            xnew, thetanew = optimize_basis(A[:, active], x[active], theta[active], fs)

            x[active] = xnew
            theta[active] = thetanew
            active[active] = np.logical_not(np.isclose(xnew, 0))

            logging.debug("x %s" % x)

            # check optimality
            if np.allclose(fs["optimal_nz"](A[:, active], x[active]), 0):
                if not np.all(fs["optimal_z"](A[:, np.logical_not(active)], x[np.logical_not(active)]) <= gamma):
                    # let another variable enter
                    break
                else:
                    # optimal
                    return x

def optimize_basis(A, x0, theta, fs):
    x1 = fs["qp_optimum"](A, theta)
    logging.debug("optimum %s" % x1)

    # find zero-crossings
    betas = x0 / (x0 - x1)
    betas = betas[np.logical_and(0 <= betas, betas < 1)]
    # make sure we investigate x1
    betas = np.append(betas, 1)
    logging.debug("betas %s xs %s" % (betas, fs["txs"](betas, x0, x1)))

    x = fs["select_candidate"](A, betas, x0, x1)
    theta = np.sign(x)

    return x, theta


l1ls_featuresign(np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]), np.array([0.0, 1.0, 2.0]), 0.0)
