import torch
from torch.func import grad
from torch.func import vmap
from torch.func import jacrev
from torch.func import hessian
import cvxpy as cp
from cvxpylayers.torch.cvxpylayer import CvxpyLayer
#from cvxpylayers_mod.cvxpylayers.torch.cvxpylayer import CvxpyLayer
import Prox
import math
import Metric as mx

import matplotlib.pyplot as plt
from torch.func import vmap

import pickle

from pathlib import Path

def semidef_cone_eig_prox(X,C):
    lmda,V = torch.linalg.eigh(X)
    lmda = torch.real(lmda)
    V = torch.real(V)
    lmda_new = torch.clamp(lmda,0)
    return V@torch.diag(lmda_new)@V.T


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

def get_semidef_cone_prox(N, P = None):
    if P == None:
        P = torch.ones(N,N)
    gamma = 1.0
    X = cp.Variable((N,N))
    Z = cp.Parameter((N,N))
    constraints = [  X >> 0  ]
    problem  = cp.Problem(cp.Minimize( cp.sum(  cp.multiply(P, (X-Z)**2)  )  ),  constraints)   #  Correct one

    qp_cvxlayer = CvxpyLayer(problem, parameters=[Z], variables=[X])
    qp_cvxlayer_post = lambda z: qp_cvxlayer(z)[0]
    return qp_cvxlayer_post


def get_parametric_semidef_cone_prox(N):
    gamma = 1.0
    X = cp.Variable((N,N))
    PZ = cp.Parameter((N,N))
    P = cp.Parameter((N,N))
    constraints = [  X >> 0  ]
    #problem  = cp.Problem(cp.Minimize( cp.sum(  cp.multiply(P, (X-Z)**2)  )  ),  constraints)
    problem  = cp.Problem(cp.Minimize( cp.sum(  cp.multiply(P, X**2)  )  - 2.0*cp.sum(  cp.multiply(PZ,X)  )  ),  constraints)
    qp_cvxlayer = CvxpyLayer(problem, parameters=[PZ,P], variables=[X], gp=True)
    qp_cvxlayer_post = lambda PZ,P: qp_cvxlayer(PZ,P)[0]
    return qp_cvxlayer_post


def get_parametric_semidef_cone_prox_temp(N):
    gamma = 1.0
    X = cp.Variable((N,N))
    P = cp.Parameter((N,N))
    constraints = [  X >> 0  ]
    #problem  = cp.Problem(cp.Minimize( cp.sum(  cp.multiply(P, (X-Z)**2)  )  ),  constraints)
    problem  = cp.Problem(cp.Minimize( cp.sum(  cp.multiply(P, X**2)  )  ),  constraints)
    qp_cvxlayer = CvxpyLayer(problem, parameters=[P], variables=[X])
    qp_cvxlayer_post = lambda PZ,P: qp_cvxlayer(P)[0]
    return qp_cvxlayer_post


# Baseline cvxpy solver
def get_SDP_solver(N):
    X = cp.Variable((N,N))
    C = cp.Parameter((N,N))
    constraints = [  cp.trace(X) == 1, X>>0  ]
    problem  = cp.Problem(cp.Minimize( cp.trace(C@X)  ),  constraints)   #  c @ x  +  (1.0/2.0*gamma)*cp.norm(x-z)**2
    qp_cvxlayer = CvxpyLayer(problem, parameters=[C], variables=[X])
    qp_cvxlayer_post = lambda C: qp_cvxlayer(C)[0]
    return qp_cvxlayer_post

# Our custom solver
def SDP_DR_solve(prox_eq_lp,prox_cone,C,Z,N,n_steps):
    batch = C.shape[0]
    X = Z
    for _ in range(n_steps):
        Y = prox_cone(X,C)
        Z = prox_eq_lp(2*Y.view(batch,-1) - X.view(batch,-1), C.view(batch,-1)).view(batch,N,N)
        X = X + (Z-Y)
    return prox_cone(X,C)

# minimize (x-z)^T P (x-z)
# Project Z onto the semidefinite cone wrt metric P
def parametric_semidef_cone_prox_pgd(P,Z,projection,N,n_steps,alpha):
    batch = Z.shape[0]
    X = Z
    C = torch.zeros(X.shape) # dummy input
    for _ in range(n_steps):
        grad = (  2*P.view(batch,-1)*(X.view(batch,-1) - Z.view(batch,-1))  ).view(batch,N,N)
        X = projection( X - alpha*grad, C )
    return X





N = 5

def f_obj(x,c):
    x = x[:N**2]
    return c@x          #lambd * torch.sum(x*torch.mv(Q,x))   #torch.sum( -p*x  ) +          #-p@x + lambd * x@(Q@x)

'''
# DEFINE THE CONSTRAINTS
'''
I = torch.eye(N); i = I.flatten()
def F_eq(x,p):
    #x = x[:x_dim]   # careful: n_dim becomes x_dim + n_ineq and x_dim becomes n_dim
    return (i@x - 1.0).unsqueeze(0)

def F(xs, parms):
    x = xs[0:N**2]
    return F_eq(xs,parms)

Pm_Id = mx.Identity(N**2,N**2)
prox_custom_a = Prox.QuadraticObjectiveLinearEqualityComposition(f_obj, F, Pm_Id, n_dim = N**2, parm_dim = N**2)
prox_custom_b = Prox.EqualityConstrainedQuadratic(f_obj, F, Pm_Id, n_dim = N**2, parm_dim = N**2, JF_fixed=False, gamma=1.0)

SDP_solver = get_SDP_solver(N)



torch.manual_seed(0)


#prox_cvx = get_equality_lp_prox(N)
prox_cvx = get_equality_lp_prox_vec(N**2)
R = torch.rand(N,N);  Z = R + R.T
R = torch.rand(N,N);  C = R + R.T
R = torch.rand(N,N);  A = R + R.T
R = torch.rand(N,N);  P = R + R.T


"""
Compare the outputs of the individual proxes
"""
#sol_cvx = prox_cvx(Z,C)
prox_out_cvx = prox_cvx( Z.flatten(), C.flatten() ).view(N,N)
prox_out_custom_a = prox_custom_a( Z.flatten().unsqueeze(0), C.flatten().unsqueeze(0) ).view(N,N)
prox_out_custom_b = prox_custom_b( Z.flatten().unsqueeze(0), C.flatten().unsqueeze(0) ).view(N,N)
print("prox_out_cvx")
print( prox_out_cvx )
print("prox_out_custom_a")
print( prox_out_custom_a )
print("prox_out_custom_b")
print( prox_out_custom_b )
print("torch.diag(prox_out_cvx).sum()")
print( torch.diag(prox_out_cvx).sum() )
print("torch.diag(prox_out_custom_a).sum()")
print( torch.diag(prox_out_custom_a).sum() )
print("torch.diag(prox_out_custom_b).sum()")
print( torch.diag(prox_out_custom_b).sum() )
cone_prox = get_semidef_cone_prox(N)
cone_prox_out = cone_prox(Z)
#param_cone_prox = get_parametric_semidef_cone_prox_temp(N)    # DPP error
#param_cone_prox_out = param_cone_prox(P)
#print("param_cone_prox_out")
#print( param_cone_prox_out )

"""
Compare the outputs of the full SDP solver vs cvxpy on a batch input
"""
batch = 2
C = C.repeat(batch,1,1)
Z = Z.repeat(batch,1,1)
n_steps = 100

prox_eq_lp = Prox.EqualityConstrainedQuadratic(f_obj, F, Pm_Id, n_dim = N**2, parm_dim = N**2, JF_fixed=False, gamma=1.0)
prox_cone  = get_semidef_cone_prox(N)
prox_cone_eig = vmap(semidef_cone_eig_prox)
prox_cone_out = prox_cone(Z)
prox_cone_eig_out = prox_cone_eig(Z,C)
print("prox_cone_out")
print( prox_cone_out )
print("prox_cone_eig_out")
print( prox_cone_eig_out )
sol_cvx = SDP_solver(C)
sol_custom = SDP_DR_solve(prox_eq_lp,prox_cone_eig,C,Z,N,n_steps)
print("sol_cvx")
print( sol_cvx )
print("sol_custom")
print( sol_custom )
# Variable metric cone projections
n_steps_proj = 50
alpha_proj = 0.05
prox_cone_var = get_semidef_cone_prox(N,P)
metric = mx.FlatIdentity(N**2,N**2,P_d = P.flatten()) #mx.ParametricDiagonal(N**2,N**2,10.0,1.0/10.0)   # TODO watch out for ub, lb
module_parametric_semidef_cone_prox_pgd = Prox.SemidefiniteConePGD(metric, n_steps = n_steps_proj, stepsize = alpha_proj)

prox_cone_var_out = prox_cone_var(Z)
prox_cone_var_pgd_out = parametric_semidef_cone_prox_pgd(P.repeat(batch,1,1),Z,prox_cone_eig,N,n_steps_proj,alpha_proj)
module_prox_cone_var_pgd_out = module_parametric_semidef_cone_prox_pgd(Z,C)    # this one takes a dummy copy of the problem params


print("prox_cone_var_out")
print( prox_cone_var_out )
print("prox_cone_var_pgd_out")
print( prox_cone_var_pgd_out )
print("module_prox_cone_var_pgd_out")
print( module_prox_cone_var_pgd_out )




"""
Training Data
"""






n_samples = 250
batch  = 25
epochs = 2000
n_steps = 10

torch.manual_seed(0)

filename = Path("./SDP_data/n_samples{}_batch{}_epochs{}_n_steps{}.p".format(n_samples, batch, epochs, n_steps))
if filename.exists():
    print("Parameter data file already exists")
    (R,Z,C,X_opt,X_init) = pickle.load(open(filename,'rb'))
else:
    print("Generating parameter data")
    R = torch.rand(N,N);            Z = (R + R.T).repeat(n_samples,1,1)  # initial iterates
    R = torch.rand(n_samples,N,N);  C =  R + R.permute(0,2,1)         # problem data
    X_opt = SDP_solver(C)
    X_init = X_opt + 0.25*Z
    pickle.dump( (R,Z,C,X_opt,X_init), open(filename,'wb') )

#R = torch.rand(N,N);            Z = (R + R.T).repeat(n_samples,1,1)  # initial iterates
#R = torch.rand(n_samples,N,N);  C =  R + R.permute(0,2,1)         # problem data
#X_opt = SDP_solver(C)
#X_init = X_opt + 0.25*Z

eval_interval = 10

Z_test = Z[ int(n_samples*0.8):]
Z      = Z[:int(n_samples*0.8) ]
C_test = C[ int(n_samples*0.8):]
C      = C[:int(n_samples*0.8) ]
X_opt_test  =  X_opt[ int(n_samples*0.8):]
X_opt       =  X_opt[:int(n_samples*0.8) ]
X_init_test = X_init[ int(n_samples*0.8):]
X_init      = X_init[:int(n_samples*0.8) ]



"""
Baseline Test (Euclidean Prox)
"""
baseline_steps = [5,10,15,20,30]
metric_eq_lp = mx.Identity(N**2,N**2)
metric_cone  = mx.Identity(N**2,N**2,flat=True)
prox_eq_lp = Prox.QuadraticObjectiveLinearEqualityComposition(f_obj, F, metric_eq_lp, n_dim = N**2, parm_dim = N**2)
prox_cone  = Prox.SemidefiniteConePGD(metric_cone, n_steps = n_steps_proj, stepsize = alpha_proj)
L2_base = {}
ineq_viol_base = {}
eq_viol_base = {}
for n_steps_base in baseline_steps:
    X = SDP_DR_solve(prox_eq_lp,prox_cone,C_test,X_init_test,N,n_steps_base)
    L2 = torch.norm(X-X_opt_test,p=2,dim=1).mean().item()
    L2_base[n_steps_base] = L2
    print("L2_base_{}step = {}".format(n_steps_base,L2))
    ineq_viol = torch.relu( -torch.linalg.eigh( X ).eigenvalues ).mean(1).mean(0).item()  # each row has the eigenvalues of a matrix in X
    ineq_viol_base[n_steps_base] = ineq_viol
    eq_viol = torch.abs( torch.stack([torch.trace(z)-1.0 for z in X]) ).mean().item()
    eq_viol_base[n_steps_base]   =   eq_viol



"""
Training Experiment
"""
predictor    = torch.nn.Sequential( torch.nn.Linear(  N**2,2*N**2), torch.nn.ReLU(), torch.nn.BatchNorm1d(2*N**2),
                                    torch.nn.Linear(2*N**2,2*N**2), torch.nn.ReLU(), torch.nn.BatchNorm1d(2*N**2),
                                    torch.nn.Linear(2*N**2,  N**2)  )
metric_eq_lp = mx.ParametricIdentity(N**2,N**2,10.0,1.0/10.0)#mx.ParametricDiagonal(N**2,N**2,10.0,1.0/10.0)
metric_cone  = mx.ParametricDiagonal(N**2,N**2,1.0,1.0,flat=True)
#prox_eq_lp = Prox.EqualityConstrainedQuadratic(f_obj, F, metric, n_dim = N**2, parm_dim = N**2, JF_fixed=False, gamma=1.0)   #JK TODO: fix this to use self.metric in the prox
prox_eq_lp = Prox.QuadraticObjectiveLinearEqualityComposition(f_obj, F, metric_eq_lp, n_dim = N**2, parm_dim = N**2)
prox_cone  = Prox.SemidefiniteConePGD(metric_cone, n_steps = n_steps_proj, stepsize = alpha_proj)

sgd_params  = list(prox_eq_lp.parameters()) + list(prox_cone.parameters())
#sgd_params += list( predictor.parameters())

optimizer = torch.optim.AdamW(sgd_params, lr=1e-2)
mse = torch.nn.MSELoss()
loss_list = []
L2_list   = []
ineq_viol_list = []
eq_viol_list = []
for epoch in range(epochs):

    if epoch % eval_interval == 0:
        with torch.no_grad():
            n_test = C_test.shape[0]
            X_warm = X_init_test
            #X_warm = predictor( C_test.view(n_test,-1) ).view(n_test,N,N)

            X = SDP_DR_solve(prox_eq_lp, prox_cone, C_test, X_warm, N, n_steps) #X_warm #
            L2 = torch.norm(X-X_opt_test, p=2, dim=1).mean().item()
            L2_list.append(L2)
            print("L2 test error = {}".format(L2))
            ineq_viol = torch.relu( -torch.linalg.eigh( X ).eigenvalues ).mean(1).mean(0).item()  # each row has the eigenvalues of a matrix in X
            ineq_viol_list.append( ineq_viol )
            eq_viol = torch.abs( torch.stack([torch.trace(z)-1.0 for z in X]) ).mean().item()
            eq_viol_list.append(     eq_viol )

    print("Training epoch {}:".format(epoch))
    idx = torch.randperm(C.shape[0])[:batch]

    X_warm = X_init[idx]
    #X_warm = predictor( C[idx].view(batch,-1) ).view(batch,N,N)
    X = SDP_DR_solve(prox_eq_lp, prox_cone, C[idx], X_warm, N, n_steps) #X_warm #
    loss = mse(X.view(batch,-1),X_opt[idx].view(batch,-1))
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    loss_list.append(loss.item())




filename = 'size{}_step{}_learn_gamma_linear'.format(N,n_steps)

length = len(L2_list)
plt.semilogy(range(length), L2_list,          label = "Trainable Prox in {} steps".format(n_steps) )
for key,val in L2_base.items():
    plt.semilogy(range(length), [val]*length, color = 'grey', linestyle='dashed', label = "Euclidean in {} steps: L2 = {}".format(key,val))
plt.ylabel("L2 Error")
plt.xlabel("Epoch")
#plt.legend(loc = "lower left")
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2))
plt.tight_layout()
plt.savefig(filename+'.png', bbox_inches="tight")
plt.close()

print("ineq_viol_list")
print( ineq_viol_list )
print("eq_viol_list")
print( eq_viol_list )
print("ineq_viol_base")
print( ineq_viol_base )
print("eq_viol_base")
print( eq_viol_base )

plt.semilogy(range(len(ineq_viol_list)), ineq_viol_list,          label = "Ineq Violation after {} steps".format(n_steps) )
plt.semilogy(range(len(  eq_viol_list)),   eq_viol_list,          label = "Eq Violation after {} steps".format(n_steps) )
for key,val in ineq_viol_base.items():
    plt.semilogy(range(len(ineq_viol_list)), [val]*length, color = 'grey',  linestyle='dashed', label = "Ineq Violation after {} steps: L2 = {}".format(key,val))
for key,val in   eq_viol_base.items():
    plt.semilogy(range(len(  eq_viol_list)), [val]*length, color = 'black', linestyle='dashed', label =   "Eq Violation after {} steps: L2 = {}".format(key,val))
plt.ylabel("Average Violation")
plt.xlabel("Epoch")
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2))
plt.tight_layout()
plt.savefig(filename+'_viols.png', bbox_inches="tight")
plt.close()


print("loss_list")
print( loss_list )
print("L2_list")
print( L2_list )


print("L2_base")
print( L2_base )







out_dict = {"loss_list":loss_list, "L2_list":L2_list, "L2_baseline":L2_base}




with open(filename+'.p', 'wb') as handle:
    pickle.dump(out_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
