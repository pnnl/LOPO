import torch
import torch.nn as nn
import operator
from functools import reduce
import numpy as np
import osqp
import time
from numpy import linalg as LA

import cvxpy as cp
from cvxpylayers.torch.cvxpylayer import CvxpyLayer
from pathlib import Path
from portfolio_utils import gen_portfolio_lto_data


# minimize lambda_1(E Q E^T) / lambda_1(E Q E^T)
# to get L = (E^T E)^{-1}
# then the optimal metric is K = E^T E = L^{-1}

def optimal_metric_QP(H,A):
    Q = A@(torch.linalg.inv(H))@A.T

    n = Q.shape[0]
    L = cp.Variable(n)
    t = cp.Variable(1)
    # The operator >> denotes matrix inequality.
    constraints = [t*Q >> cp.atoms.diag(L), cp.atoms.diag(L) >> Q]

    prob = cp.Problem(cp.Minimize(t),  constraints)
    prob.solve()

    # Print result.
    print("The optimal value is", prob.value)
    print("A solution X is")
    print(L.value)

    K = torch.Tensor(  np.diag(1/L.value)  ).double()
    E = np.sqrt(K)
    M = E@Q@E.T
    lmda,v = np.linalg.eig(M)

    lmda_max = np.max(lmda)
    lmda_min = np.min(lmda)
    gamma = (1/np.sqrt(lmda_max*lmda_min))

    return K, gamma


if __name__ == "__main__":

    data_loaded = gen_portfolio_lto_data(8,1,1,1)
    Q_load = torch.Tensor(data_loaded['Q']).double()

    H = 2*Q_load
    A = torch.eye(Q_load.shape[0]).double()
    K,gamma = optimal_metric_QP(H,A)

    print("K = ")
    print( K    )
    print("gamma = ")
    print( gamma    )
