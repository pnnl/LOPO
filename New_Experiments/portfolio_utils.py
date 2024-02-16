import torch
import torch.nn as nn
import operator
from functools import reduce
import numpy as np
import osqp
#import cyipopt     # JK to suppress cyiopt not installed - version conflict (requires python 3.11, pytorch not compatible)
from scipy.sparse import csc_matrix
import time
from numpy import linalg as LA

import cvxpy as cp
from cvxpylayers.torch.cvxpylayer import CvxpyLayer

# This function constructs a portfolio problem (specifically its covariance matrix)
# Along with a set of training data (historical asset prices)

def param_gen(n_dim,ntrain,nvalid,ntest):

    idio_risk = 1e-2 #5e-3
    alpha = 0.024
    scale_factor = 100.0 #00.0
    noise_var = .01 #.02

    N = ntrain+nvalid+ntest
    ret_cov_np = f"portfolio_data/eod_ret_cov_factor.npz"
    ret_cov_loaded = np.load(ret_cov_np)   ### sigma, return covariance matrix
    cov = scale_factor * (ret_cov_loaded['cov'][:n_dim, :n_dim] + idio_risk * np.eye(n_dim))
    ret = ret_cov_loaded['ret'][1:,:n_dim]

    condition_number = 1
    #condition_number = LA.cond(cov)  ## the conditioning number should be small


    mu_mat = np.zeros((N, n_dim))
    T = ret.shape[0]
    noise = np.sqrt(noise_var) * np.random.normal(size=(N, n_dim))

    for i in range(N):
        time_index = i % T
        mu_mat[i, :] = scale_factor * alpha * (ret[time_index, :] + noise[i, :])

    return mu_mat, cov, condition_number


def gen_portfolio_lto_data(n_dim,ntrain,nvalid,ntest):

    mu, COV, conditioning_number = param_gen(n_dim,ntrain,nvalid,ntest)

    #ntrain = 10000
    #nvalid = 1000
    #ntest = 1000
    nex = ntrain + nvalid + ntest  # the total number of instances
    neq = 1  # the number of equality constraints
    nineq = n_dim  # the number of inequality constraints

    np.random.seed(1001)

    Q = COV * 2

    p = -mu.T
    A = np.ones((1, n_dim))
    h = np.zeros(n_dim)
    G = -np.eye(n_dim)

    print("Generating dataset . . . ")

    x = np.ones((1, ntrain)).T
    p_train = p[:, :ntrain]
    #train_Y, t = solve_convexqp(Q, p_train, A, G, h, x)
    #print(train_Y[0,:])
    #print(train_Y[1, :])

    print("Training set generated ")

    x = np.ones((1, nvalid)).T
    p_valid = p[:, ntrain : ntrain + nvalid]
    #valid_Y, t = solve_convexqp(Q, p_valid, A, G, h, x)

    print("Validation set generated ")

    x = np.ones((1, ntest)).T
    p_test = p[:, ntrain + nvalid :]
    #test_Y = solve_convexqp(Q, p_test, A, G, h, x)

    print("Test set generated ")

    filename = "./portfolio_data/portfolio_var{}_train{}_valid{}_test{}.npz".format(n_dim, ntrain, nvalid, ntest)
    with open(filename, 'wb') as f:
        #np.savez_compressed(f, Q=Q, A=A, G=G, h=h, x=x, trainX=p_train.T, validX=p_valid.T, testX=p_test.T, trainY=train_Y, validY=valid_Y, testY=test_Y)
        np.savez_compressed(f, Q=Q, A=A, G=G, h=h, x=x, trainX=p_train.T, validX=p_valid.T, testX=p_test.T, trainY=0, validY=0, testY=0)

    data_loaded = np.load(filename, allow_pickle=True)
    return data_loaded



###################################################################
# CONVEX QP PROBLEM
###################################################################
def solve_convexqp(Q, p, A, G, h, X, tol=1e-4):
    #print('running osqp')
    ydim = Q.shape[0]
    my_A = np.vstack([A, G])
    total_time = 0
    Y = []
    #Xi = np.array(1)
    Xi = np.array(X)
    avg_obj = 0
    for pi in p.T:
        """
                OSQP solver problem of the form

                minimize     1/2 x' * P * x + q' * x
                subject to   l <= A * x <= u

                solver settings can be specified as additional keyword arguments
        """
        solver = osqp.OSQP()
        my_l = np.hstack([Xi, -np.ones(h.shape[0]) * np.inf])
        my_u = np.hstack([Xi, h])
        solver.setup(P=csc_matrix(Q), q=pi, A=csc_matrix(my_A), l=my_l, u=my_u, verbose=False, eps_prim_inf=tol)
        start_time = time.time()
        results = solver.solve()
        end_time = time.time()
        total_time += (end_time - start_time)
        avg_obj += .5 * (results.x @ Q) @ results.x + pi @ results.x
        #print(.5 * (results.x @ Q) @ results.x)
        #print(pi @ results.x)
        if results.info.status == 'solved':
            Y.append(results.x)
        else:
            Y.append(np.ones(ydim) * np.nan)
        sols = np.array(Y)

    return sols , total_time #, avg_obj/p.shape[0] #, total_time/X.shape[0]


# Solution from CVXPY
def cvx_qp(N,Q,budget):
	x = cp.Variable(N)
	c = cp.Parameter(N)
	constraints = [  0<=x, cp.sum(x) == budget  ]
	problem  = cp.Problem(cp.Maximize(  c @ x  - cp.quad_form(x,Q)   ),  constraints)
	qp_cvxlayer = CvxpyLayer(problem, parameters=[c], variables=[x])
	qp_cvxlayer_post = lambda z: qp_cvxlayer(z)[0]
	return qp_cvxlayer_post
