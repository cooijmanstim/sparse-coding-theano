import os
import numpy as np
import theano
import theano.tensor as T
import theano.sandbox.linalg as tl
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
    
    error = lambda x: ((T.dot(tA, x) - ty)**2).sum()
    derror = lambda x: T.grad(error(x), x)
    penalty = lambda x: gamma*abs(x).sum()
    loss = lambda x: error(x) + penalty(x)
    loss_if_consistent = lambda x: error(x) + gamma*T.dot(ttheta, x)

    entering_index = T.argmax(abs(derror(tx)))
    txs, _ = theano.map(lambda b, x0, x1: (1-b)*x0 + b*x1,
                        [tbetas], [tx0, tx1])

    errors = theano.map(error, [txs])[0]
    losses = theano.map(loss, [txs])[0]

    return {
        "loss": theano.function([tA, tx], loss(tx),
                                givens = {ty: y}),
        "loss_if_consistent": theano.function([tA, tx, ttheta],
                                              loss_if_consistent(tx),
                                              givens = {ty: y}),
        "select_entering": theano.function([tA, tx],
                                           [entering_index, derror(tx)[entering_index]],
                                           givens = {ty: y}),
        "qp_optimum": theano.function([tA, ttheta],
                                      T.dot(tl.matrix_inverse(T.dot(tA.T, tA)), T.dot(tA.T, ty) - gamma/2*ttheta),
                                      givens = {ty: y}),
        "qp_candidates": theano.function([tA, tbetas, tx0, tx1],
                                         [txs, losses],
                                         givens = {ty: y}),
        "select_candidate": theano.function([tA, tbetas, tx0, tx1],
                                            txs[T.argmin(losses)],
                                            givens = {ty: y}),
        "optimal_nz": theano.function([tA, tx],
                                      derror(tx) + gamma*T.sgn(tx),
                                      givens = {ty: y}),
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
        # select entering variable
        zero_mask = x == 0
        xz = x[zero_mask]
        iz, l = fs["select_entering"](A[:, zero_mask], xz)
        # iz is an index into xz; figure out the corresponding index into x
        i = np.where(zero_mask)[0][iz]

        if abs(l) > gamma:
            theta[i] = -np.sign(l)
            active[i] = True
            basis_optimal = False
            logging.debug("enter %i, grad %f, gamma %f, sign %i" % (i, l, gamma, theta[i]))

        logging.debug("x %s theta %s" % (x, theta))

        if basis_optimal:
            logging.debug("optimal")
            break

        while not basis_optimal:
            # optimize active variables
            xnew, thetanew = optimize_basis(A[:, active], x[active], theta[active], fs)

            x[active] = xnew
            theta[active] = thetanew
            active[active] = np.logical_not(np.isclose(xnew, 0))

            logging.debug("x %s theta %s active %s" % (x, theta, active))

            # check optimality
            optimal_nz = fs["optimal_nz"](A[:, active], x[active])
            logging.debug("optimal_nz %s" % optimal_nz)
            if np.allclose(optimal_nz, 0):
                # maybe let another variable enter
                basis_optimal = True

def optimize_basis(A, x0, theta, fs):
    x1 = fs["qp_optimum"](A, theta)
    lossc0, lossc1 = map(lambda x: fs["loss_if_consistent"](A, x, theta), [x0, x1])
    loss0, loss1 = map(lambda x: fs["loss"](A, x), [x0, x1])
    logging.debug("qp_optimum x0 %s lossc0 %f loss0 %f" % (x0, lossc0, loss0))
    logging.debug("qp_optimum x1 %s lossc1 %f loss1 %f" % (x1, lossc1, loss1))

    # find zero-crossings
    betas = x0 / (x0 - x1)
    betas = betas[np.logical_and(0 < betas, betas < 1)]
    # make sure we investigate x1
    betas = np.append(betas, 1)

    xs, losses = fs["qp_candidates"](A, betas, x0, x1)
    logging.debug("betas %s xs %s losses %s" % (betas, xs, losses))

    x = xs[np.argmin(losses)]
    logging.debug("selected candidate %s" % x)

    theta = np.sign(x)
    return x, theta
