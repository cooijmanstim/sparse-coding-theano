def sparse_coding(X, num_bases, beta, num_iters):
    B = np.random((X.shape[0], num_bases)) - 0.5;
    B = B / np.sqrt(np.sum(B**2, 0))

    S = np.zeros((num_bases, X.shape[1]))

    for t in xrange(num_iters):
        # shuffle samples
        np.random.shuffle(X.T)

        # TODO: this function uses the other convention, where rows are samples.
        # use that one everywhere.  also, figure out how i was expecting to deal
        # with multiple samples in there.
        S = l1ls_featuresign(B, X, beta/sigma*noise_var, S);
        S[np.isnan(S)] = 0;

        l2ls_learn_basis_dual(X, S, 1);
