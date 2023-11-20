# Import packages.
import cvxpy as cp
import numpy as np
from cvxpylayers.torch.cvxpylayer import CvxpyLayer
import torch
from torch.func import vmap
import matplotlib.pyplot as plt
import Nets

# Generate a random non-trivial quadratic program.
m = 15
n = 10
p = 5
np.random.seed(1)
C = torch.Tensor(  np.random.randn(n, n)   )
C = torch.Tensor(  C.T @ C                 )
q = torch.Tensor(  np.random.randn(n)      )
G = torch.Tensor(  np.random.randn(m, n)   )
h = torch.Tensor(  G @ np.random.randn(n)  )
A = torch.Tensor(  np.random.randn(p, n)   )
b = torch.Tensor(  np.random.randn(p)      )

n_samples = 5000
batch = 100
#q_train = torch.Tensor(  np.random.randn(n_samples,n)      )
q_train = torch.Tensor(  10*(np.random.randn(n_samples,n) - 0.5)      )


def f(x,q):
    return (1/2)*x@(C@x) + q@x

def L(x,lmda,nu,q):
    return (1/2)*x@(C@x) + q@x - lmda@x + nu@(A@x - b)

def eq_viol(x):
    return torch.abs(A @ x - b).mean()

def ineq_viol(x):
    return torch.relu(-x).mean()

# Define and solve the PRIMAL problem.
x = cp.Variable(n)
primal = cp.Problem(cp.Minimize((1/2)*cp.quad_form(x, C) + q @ x),
                 [ A @ x == b,  x >= 0 ])
primal.solve()

def get_primal():
    x = cp.Variable(n)
    q = cp.Parameter(n)
    problem  = cp.Problem(cp.Minimize((1/2)*cp.quad_form(x, C) + q @ x),
                     [ A @ x == b,  x >= 0  ])
    qp_cvxlayer = CvxpyLayer(problem, parameters=[q], variables=[x])
    qp_cvxlayer_post = lambda q: qp_cvxlayer(q)[0]
    return qp_cvxlayer_post





# Define and solve the DUAL problem.
#u = cp.Variable(n)
#v = cp.Variable(p)
#dual = cp.Problem(cp.Maximize(-(1/2)*cp.quad_form(u, C) + b @ v),
#                 [ A.T @ v - C @ u <= p])
#dual.solve()
def get_dual():
    u = cp.Variable(n)
    v = cp.Variable(p)
    q = cp.Parameter(n)
    problem  = cp.Problem(cp.Maximize(-(1/2)*cp.quad_form(u, C) + b @ v),
                     [ A.T @ v - C @ u <= q])
    qp_cvxlayer = CvxpyLayer(problem, parameters=[q], variables=[u,v])
    qp_cvxlayer_post = lambda q: qp_cvxlayer(q)[0]
    return qp_cvxlayer_post


def cvx_lagr(n,p):
    x     = cp.Variable(n)
    lmda  = cp.Parameter(n)
    nu    = cp.Parameter(p)
    q     = cp.Parameter(n)
    lagr  = cp.Problem(cp.Minimize((1/2)*cp.quad_form(x, C) + q @ x - lmda@x + nu@(A@x - b)),
                     [ ])
    lagr_cvxlayer = CvxpyLayer(lagr, parameters=[q,lmda,nu], variables=[x])
    lagr_cvxlayer_post = lambda q,lmda,nu: lagr_cvxlayer(q,lmda,nu)[0]
    return lagr_cvxlayer_post


def cvx_qp(N,Q):
    x = cp.Variable(N)
    p = cp.Parameter(N)
    z = cp.Parameter(N)
    constraints = [  cp.sum(x) == 1  ]
    problem  = cp.Problem(cp.Minimize(  p @ x  + (1.0/2.0)*cp.quad_form(x,Q) + (1.0/2.0*gamma)*cp.quad_form(x-z,I)   ),  constraints)  #
    qp_cvxlayer = CvxpyLayer(problem, parameters=[p,z], variables=[x])
    qp_cvxlayer_post = lambda p,z: qp_cvxlayer(p,z)[0]
    return qp_cvxlayer_post



#x = cp.Variable(n)
#prob = cp.Problem(cp.Minimize((1/2)*cp.quad_form(x, P) + q.T @ x),
#                 [G @ x <= h,
#                  A @ x == b])
#prob.solve()

# Print result.

torch.manual_seed(0)

x_opt    = torch.Tensor( x.value  )
nu_opt   = torch.Tensor( primal.constraints[0].dual_value  )
lmda_opt = torch.Tensor( primal.constraints[1].dual_value  )

primal = get_primal()
x_opt_cvxlayer = primal(q)

lagr = cvx_lagr(n,p)
x_lagr_opt = lagr(q,lmda_opt,nu_opt)
f_x_lagr_opt = f(x_lagr_opt,q)

print("x_opt = ")
print( x_opt    )
print("x_opt_cvxlayer = ")
print( x_opt_cvxlayer    )
print("nu_opt = ")
print( nu_opt    )
print("lmda_opt = ")
print( lmda_opt    )
print("x_lagr_opt = ")
print( x_lagr_opt    )
print("f(x_lagr_opt,C,q) = ")
print( f_x_lagr_opt     )

for _ in range(100):
    lmda     = torch.relu( torch.rand(lmda_opt.shape) )
    nu       = torch.rand( nu_opt.shape )
    x_lagr   = lagr(q,lmda,nu)
    f_x_lagr = f(x_lagr,q)
    #print("lmda = {}".format(lmda))
    print("f_x_lagr - f_x_lagr_opt = {}".format(f_x_lagr - f_x_lagr_opt))


"""
Precompute optimal solutions for training metric
"""
x_opt_train = primal(q_train)

print("x_opt_train")
for i in range(len(x_opt_train)):
    print( x_opt_train[i] )
input("waiting")

"""
Training
"""
#predictor  = torch.nn.Sequential( torch.nn.Linear(  n,2*n), torch.nn.ReLU(), torch.nn.BatchNorm1d(2*n),
#                                  torch.nn.Linear(2*n,2*n), torch.nn.ReLU(), torch.nn.BatchNorm1d(2*n),
#                                  torch.nn.Linear(2*n,  n+p)  )
#predictor.apply(Nets.init_weights)


predictor = Nets.ReLUnet(n,n+p, hidden_sizes = [2*n,n+p], batch_norm = True, initialize = True)

optimizer = torch.optim.AdamW(predictor.parameters(), lr=1e-3)

L_list = []
error_list = []
opt_gap_list = []
eq_viol_list = []
ineq_viol_list = []
for epoch in range(10000):
    idx = torch.randperm(q.shape[0])[:batch]
    q = q_train[idx]

    duals = predictor(q)
    lmda = torch.relu( duals[:,:n ] )
    nu   = duals[:, n:]

    x  = lagr(q,lmda,nu)


    loss = ( -vmap(L)(x,lmda,nu,q) ).mean(0)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print("L = {}".format(-loss.item()))
    L_list.append(-loss.item())
    with torch.no_grad():
        x_opt_train_batch = x_opt_train[idx]
        error = torch.norm(x-x_opt_train_batch,dim=1).mean()
        print("L2 error = {}".format(error.item()))
        error_list.append(error.item())
        opt_gap = (  (vmap(f)(x,q) - vmap(f)(x_opt_train_batch,q))/vmap(f)(x_opt_train_batch,q)  ).mean()
        print("Opt gap = {} %".format(opt_gap.item()))
        opt_gap_list.append( opt_gap.item() )
        ev = vmap(  eq_viol)(x).mean()
        print("Eq Viol = {}".format(ev.item()))
        iv = vmap(ineq_viol)(x).mean()
        print("Ineq Viol = {}".format(iv.item()))
        eq_viol_list.append(ev)
        ineq_viol_list.append(iv)

plt.semilogy( range(len(error_list)), error_list, label = 'L2 Error' )
#plt.plot( range(len(L_list)), L_list, label = 'Lagrangian Dual Fn Value' )
plt.semilogy( range(len(opt_gap_list)), opt_gap_list, label = 'Optimality gap (%)' )
plt.semilogy( range(len(eq_viol_list)), eq_viol_list, label = 'Eq Viol' )
plt.semilogy( range(len(ineq_viol_list)), ineq_viol_list, label = 'Ineq Viol' )
#plt.ylim(-1,1)
plt.legend()
plt.show()

quit()

print("====PRIMAL Problem=====")
print("\nThe optimal value is", primal.value)
print("A solution x is")
print(x.value)
print("A dual solution corresponding to the inequality constraints is")
print(primal.constraints[0].dual_value)
print("A dual solution corresponding to the equality constraints is")
print(primal.constraints[1].dual_value)



print("====DUAL Problem=====")
print("\nThe optimal value is", primal.value)
print("A solution u is")
print(u.value)
print("A solution v is")
print(v.value)
print("A dual solution corresponding to the inequality constraints is")
print(dual.constraints[0].dual_value)
