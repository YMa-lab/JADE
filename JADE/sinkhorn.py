import numpy as np
import torch
import torch.nn as nn
import warnings

class NonSquareSinkhornKnopp:
    def __init__(self, max_iter=10, epsilon=1e-3):
        
        self._epsilon = epsilon
        self._max_iter = max_iter
        self._stopping_condition = None
        self._iterations = 0
        self._D1 = None
        self._D2 = None

    def fit(self, P):
        n, m = P.shape
        P = P + 1e-18
        max_thresh = 1 + self._epsilon
        min_thresh = 1 - self._epsilon
        for iter in range(self._max_iter):
            P = P / P.sum(dim=0, keepdim=True) ## col normalization
            P = P / P.sum(dim=1, keepdim=True) ## row normalization
        
        return P / n

class SinkhornKnopp:
    """
    Sinkhorn Knopp Algorithm

    Takes a non-negative square matrix P, where P =/= 0
    and iterates through Sinkhorn Knopp's algorithm
    to convert P to a doubly stochastic matrix.
    Guaranteed convergence if P has total support.

    For reference see original paper:
        http://msp.org/pjm/1967/21-2/pjm-v21-n2-p14-s.pdf

    Parameters
    ----------
    max_iter : int, default=1000
        The maximum number of iterations.

    epsilon : float, default=1e-3
        Metric used to compute the stopping condition,
        which occurs if all the row and column sums are
        within epsilon of 1. This should be a very small value.
        Epsilon must be between 0 and 1.

    Attributes
    ----------
    _max_iter : int, default=1000
        User defined parameter. See above.

    _epsilon : float, default=1e-3
        User defined paramter. See above.

    _stopping_condition: string
        Either "max_iter", "epsilon", or None, which is a
        description of why the algorithm stopped iterating.

    _iterations : int
        The number of iterations elapsed during the algorithm's
        run-time.

    _D1 : 2d-array
        Diagonal matrix obtained after a stopping condition was met
        so that _D1.dot(P).dot(_D2) is close to doubly stochastic.

    _D2 : 2d-array
        Diagonal matrix obtained after a stopping condition was met
        so that _D1.dot(P).dot(_D2) is close to doubly stochastic.

    Example
    -------

    .. code-block:: python
        >>> import numpy as np
        >>> from sinkhorn_knopp import sinkhorn_knopp as skp
        >>> sk = skp.SinkhornKnopp()
        >>> P = [[.011, .15], [1.71, .1]]
        >>> P_ds = sk.fit(P)
        >>> P_ds
        array([[ 0.06102561,  0.93897439],
           [ 0.93809928,  0.06190072]])
        >>> np.sum(P_ds, axis=0)
        array([ 0.99912489,  1.00087511])
        >>> np.sum(P_ds, axis=1)
        array([ 1.,  1.])

    """

    def __init__(self, max_iter=1000, epsilon=1e-3):
        assert isinstance(max_iter, int) or isinstance(max_iter, float),\
            "max_iter is not of type int or float: %r" % max_iter
        assert max_iter > 0,\
            "max_iter must be greater than 0: %r" % max_iter
        self._max_iter = int(max_iter)

        assert isinstance(epsilon, int) or isinstance(epsilon, float),\
            "epsilon is not of type float or int: %r" % epsilon
        assert epsilon > 0 and epsilon < 1,\
            "epsilon must be between 0 and 1 exclusive: %r" % epsilon
        self._epsilon = epsilon

        self._stopping_condition = None
        self._iterations = 0
        self._D1 = np.ones(1)
        self._D2 = np.ones(1)

    def fit(self, P):
        """Fit the diagonal matrices in Sinkhorn Knopp's algorithm

        Parameters
        ----------
        P : 2d array-like
        Must be a square non-negative 2d array-like object, that
        is convertible to a numpy array. The matrix must not be
        equal to 0 and it must have total support for the algorithm
        to converge.

        Returns
        -------
        A double stochastic matrix.

        """
        P = np.asarray(P)
        assert np.all(P >= 0)
        assert P.ndim == 2
        assert P.shape[0] == P.shape[1]

        N = P.shape[0]
        max_thresh = 1 + self._epsilon
        min_thresh = 1 - self._epsilon

        # Initialize r and c, the diagonals of D1 and D2
        # and warn if the matrix does not have support.
        r = np.ones((N, 1))
        pdotr = P.T.dot(r)
        total_support_warning_str = (
            "Matrix P must have total support. "
            "See documentation"
        )
        if not np.all(pdotr != 0):
            warnings.warn(total_support_warning_str, UserWarning)

        c = 1 / pdotr
        pdotc = P.dot(c)
        if not np.all(pdotc != 0):
            warnings.warn(total_support_warning_str, UserWarning)

        r = 1 / pdotc
        del pdotr, pdotc

        P_eps = np.copy(P)
        while np.any(np.sum(P_eps, axis=1) < min_thresh) \
                or np.any(np.sum(P_eps, axis=1) > max_thresh) \
                or np.any(np.sum(P_eps, axis=0) < min_thresh) \
                or np.any(np.sum(P_eps, axis=0) > max_thresh):

            c = 1 / P.T.dot(r)
            r = 1 / P.dot(c)

            self._D1 = np.diag(np.squeeze(r))
            self._D2 = np.diag(np.squeeze(c))
            P_eps = self._D1.dot(P).dot(self._D2)

            self._iterations += 1

            if self._iterations >= self._max_iter:
                self._stopping_condition = "max_iter"
                break

        if not self._stopping_condition:
            self._stopping_condition = "epsilon"

        self._D1 = np.diag(np.squeeze(r))
        self._D2 = np.diag(np.squeeze(c))
        P_eps = self._D1.dot(P).dot(self._D2)

        return P_eps
    
    
    
class Sinkhorn(nn.Module):
    """
    BiStochastic Layer turns the input matrix into a bi-stochastic matrix.
    Parameter: maximum iterations max_iter
               a small number for numerical stability epsilon
    Input: input matrix s
    Output: bi-stochastic matrix s
    """
    def __init__(self, max_iter=10, epsilon=1e-4):
        super(Sinkhorn, self).__init__()
        self.max_iter = max_iter
        self.epsilon = epsilon

    def forward(self, s, nrows=None, ncols=None, exp=False, exp_alpha=20, dummy_row=False, dtype=torch.float32):
        batch_size = s.shape[0]

        if dummy_row:
            dummy_shape = list(s.shape)
            dummy_shape[1] = s.shape[2] - s.shape[1]
            s = torch.cat((s, torch.full(dummy_shape, 0.).to(s.device)), dim=1)
            new_nrows = ncols
            for b in range(batch_size):
                s[b, nrows[b]:new_nrows[b], :ncols[b]] = self.epsilon
            nrows = new_nrows

        
        row_norm_ones = torch.zeros(batch_size, s.shape[1], s.shape[1], device=s.device)  # size: row x row
        col_norm_ones = torch.zeros(batch_size, s.shape[2], s.shape[2], device=s.device)  # size: col x col
        for b in range(batch_size):
            row_slice = slice(0, nrows[b] if nrows is not None else s.shape[2])
            col_slice = slice(0, ncols[b] if ncols is not None else s.shape[1])
            row_norm_ones[b, row_slice, row_slice] = 1
            col_norm_ones[b, col_slice, col_slice] = 1

        # for Sinkhorn stacked on last dimension
        if len(s.shape) == 4:
            row_norm_ones = row_norm_ones.unsqueeze(-1)
            col_norm_ones = col_norm_ones.unsqueeze(-1)

        s += self.epsilon

        for i in range(self.max_iter):
            if exp:
                s = torch.exp(exp_alpha * s)
            if i % 2 == 1:
                # column norm
                sum = torch.sum(torch.mul(s.unsqueeze(3), col_norm_ones.unsqueeze(1)), dim=2)
            else:
                # row norm
                sum = torch.sum(torch.mul(row_norm_ones.unsqueeze(3), s.unsqueeze(1)), dim=2)

            tmp = torch.zeros_like(s)
            for b in range(batch_size):
                row_slice = slice(0, nrows[b] if nrows is not None else s.shape[2])
                col_slice = slice(0, ncols[b] if ncols is not None else s.shape[1])
                tmp[b, row_slice, col_slice] = 1 / sum[b, row_slice, col_slice]
            s = s * tmp

        if dummy_row and dummy_shape[1] > 0:
            s = s[:, :-dummy_shape[1]]

        return s