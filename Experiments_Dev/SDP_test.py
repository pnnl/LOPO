import torch
from torch.func import grad
from torch.func import vmap
from torch.func import jacrev
from torch.func import hessian
import cvxpy as cp
from cvxpylayers.torch.cvxpylayer import CvxpyLayer
import Prox
import math


def semidef_cone_eig_prox(X):
    lmda,V = torch.linalg.eig(X)
    lmda = torch.real(lmda)
    V = torch.real(V)
    lmda_new = torch.clamp(lmda,0)
    return V@torch.diag(lmda_new)@V.T

def get_semidef_cone_prox(N, P = None):
    if P == None:
        P = torch.ones(N,N)
    gamma = 1.0
    X = cp.Variable((N,N))
    Z = cp.Parameter((N,N))
    constraints = [  X >> 0  ]
    problem  = cp.Problem(cp.Minimize( cp.sum(  cp.multiply(P, (X-Z)**2)  )  ),  constraints)   #  c @ x  +  (1.0/2.0*gamma)*cp.norm(x-z)**2
    qp_cvxlayer = CvxpyLayer(problem, parameters=[Z], variables=[X])
    qp_cvxlayer_post = lambda z: qp_cvxlayer(z)[0]
    return qp_cvxlayer_post



def get_equality_lp_prox(N, P=None):
    if P == None:
        P = torch.ones(N,N)
    gamma = 1.0
    X = cp.Variable((N,N))
    C = cp.Parameter((N,N))
    Z = cp.Parameter((N,N))
    constraints = [  cp.trace(X) == 1  ]
    #problem  = cp.Problem(cp.Minimize( cp.trace(C@X)  +  (1.0/2.0*gamma)*cp.sum(  cp.multiply(P, (X-Z)**2 )  ) ),  constraints)   #  c @ x  +  (1.0/2.0*gamma)*cp.norm(x-z)**2
    problem  = cp.Problem(cp.Minimize( cp.trace(C@X)  +  (1.0/2.0*gamma)*cp.trace(  P@((X-Z)**2)   ) ),  constraints)
    qp_cvxlayer = CvxpyLayer(problem, parameters=[C,Z], variables=[X])
    qp_cvxlayer_post = lambda C,Z: qp_cvxlayer(C,Z)[0]
    return qp_cvxlayer_post



"""
A version of the above prox, for which the metric P is
treated as an input parameter; this requires a trick
to circumvent the 'DPP' requirement, which seems to prohibit
multiplying the input parameters (where we form P*(X-Z)**2)

def get_parametric_equality_lp_prox(N, P=None):
    if P == None:
        P = torch.ones(N,N)
    gamma = 1.0
    X = cp.Variable((N,N))
    C = cp.Parameter((N,N))
    Z = cp.Parameter((N,N))
    constraints = [  cp.trace(X) == 1  ]
    #problem  = cp.Problem(cp.Minimize( cp.trace(C@X)  +  (1.0/2.0*gamma)*cp.sum(  cp.multiply(P, (X-Z)**2 )  ) ),  constraints)   #  c @ x  +  (1.0/2.0*gamma)*cp.norm(x-z)**2
    #problem  = cp.Problem(cp.Minimize( cp.trace(C@X)  +  (1.0/2.0*gamma)*cp.trace(  P@((X-Z)**2)   ) ),  constraints)
    problem  = cp.Problem(cp.Minimize( cp.trace(C@X)  +  (1.0/2.0*gamma)*(cp.trace(P@(X**2)) - 2.0*cp.trace( cp.multiply(P,Z)@X ))  ),  constraints)
    qp_cvxlayer = CvxpyLayer(problem, parameters=[C,Z], variables=[X])
    qp_cvxlayer_post = lambda C,Z: qp_cvxlayer(C,Z)[0]
    return qp_cvxlayer_post
"""
def get_parametric_equality_lp_prox(N):
    gamma = 1.0
    X = cp.Variable((N,N))
    C = cp.Parameter((N,N))
    P = cp.Parameter((N,N))
    PZ = cp.Parameter((N,N)) # Assumed that this is the precomputed elementwise product of P and Z
    constraints = [  cp.trace(X) == 1  ]
    #problem  = cp.Problem(cp.Minimize( cp.trace(C@X)  +  (1.0/2.0*gamma)*cp.sum(  cp.multiply(P, (X-Z)**2 )  ) ),  constraints)   #  c @ x  +  (1.0/2.0*gamma)*cp.norm(x-z)**2
    #problem  = cp.Problem(cp.Minimize( cp.trace(C@X)  +  (1.0/2.0*gamma)*cp.trace(  P@((X-Z)**2)   ) ),  constraints)
    #problem  = cp.Problem(cp.Minimize( cp.trace(C@X)  +  (1.0/2.0*gamma)*(cp.trace(P@(X**2)) - 2.0*cp.trace( PZ@X ))  ),  constraints)
    problem  = cp.Problem(cp.Minimize( cp.trace(C@X)  +  (1.0/2.0*gamma)*(cp.trace(P@(X**2)) - 2.0*cp.trace( PZ@X ))  ),  constraints)
    qp_cvxlayer = CvxpyLayer(problem, parameters=[C,P,PZ], variables=[X])
    qp_cvxlayer_post = lambda C,P,PZ: qp_cvxlayer(C,P,PZ)[0]
    return qp_cvxlayer_post




def get_equality_lp_prox_vec(N, p = None):
    if p == None:
        p = torch.ones(N)
    gamma = 1.0
    x = cp.Variable(N)
    c = cp.Parameter(N)
    z = cp.Parameter(N)
    I = torch.eye(int(math.sqrt(N))); i = I.flatten()
    constraints = [  i@x == 1  ]
    problem  = cp.Problem(cp.Minimize( c @ x  +  (1.0/2.0*gamma)*cp.sum( cp.multiply(p, (x-z)**2) )  ),  constraints)
    qp_cvxlayer = CvxpyLayer(problem, parameters=[c,z], variables=[x])
    qp_cvxlayer_post = lambda c,z: qp_cvxlayer(c,z)[0]
    return qp_cvxlayer_post

def get_SDP_solver(N):
    X = cp.Variable((N,N))
    C = cp.Parameter((N,N))
    constraints = [  cp.trace(X) == 1, X>>0  ]
    problem  = cp.Problem(cp.Minimize( cp.trace(C@X)  ),  constraints)   #  c @ x  +  (1.0/2.0*gamma)*cp.norm(x-z)**2
    qp_cvxlayer = CvxpyLayer(problem, parameters=[C], variables=[X])
    qp_cvxlayer_post = lambda C: qp_cvxlayer(C)[0]
    return qp_cvxlayer_post

"""
C: cost parameters
Z: initial iterate
N: size of the square variable matrix
n_steps: how many DR iterations
"""
def SDP_DR_solve(C,Z,N,n_steps):

    prox_eq_lp = get_equality_lp_prox(N)
    prox_cone  = get_semidef_cone_prox(N)
    X = Z
    for _ in range(n_steps):
        Y = prox_cone(X)
        Z = prox_eq_lp(C,2*Y - X)
        X = X + (Z-Y)
    return prox_cone(X)




N = 5
R = torch.rand(N,N);  Z = R + R.T
R = torch.rand(N,N);  C = R + R.T
R = torch.rand(N,N);  A = R + R.T
R = torch.rand(N,N);  P = R + R.T
prox_eq_lp_vec = get_equality_lp_prox_vec(N**2,P.flatten())
prox_eq_lp = get_equality_lp_prox(N,P)
prox_cone  = get_semidef_cone_prox(N)
prox_cone_varP  = get_semidef_cone_prox(N,P)
solve_SDP  = get_SDP_solver(N)

"""
Parametric layers (P is a parameter)
"""
param_prox_eq_lp = get_parametric_equality_lp_prox(N)



print("C")
print( C )
print("A")
print( A )
print("Z")
print( Z )

sol = prox_eq_lp_vec(C.flatten(),Z.flatten())
S = sol.view(N,N)
print("S")
print( S )

PZ = P*Z
S = param_prox_eq_lp(C,P,PZ)#,P.flatten())
print("S")
print( S )

S = prox_eq_lp(C,Z)
print("S")
print( S )

proj = prox_cone(S)
print("proj")
print( proj )

proj = semidef_cone_eig_prox(S)
print("proj")
print( proj )

proj_varP = prox_cone_varP(S)
print("proj_varP")
print( proj_varP )




sol = solve_SDP(C)
print("sol")
print( sol )

n_steps = 200
sol = SDP_DR_solve(C,Z,N,n_steps)
print("sol")
print( sol )
