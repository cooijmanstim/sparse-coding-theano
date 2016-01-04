sparse-coding-theano
====================

Theano-based implementation of the efficient sparse-coding algorithms by Honglak Lee et al. (2006)

WARNING: This is the first thing I ever did in Theano and it could be done much better. This will be much slower than a numpy implementation as it uses several Theano functions, recompiles them at every iteration, etc. Also I never got around to testing it against Lee et al's MATLAB implementation which is definitely something you'll want to do before relying on this.
