"""
Learning to solve parametric Quadratic Programming
portfolio optimization
problems using Neuromancer.

Problem formulation:
    minimize    - p^T x + lambda x^T Q x
    subject to       1^T x = 1
                      x >= 0

Where p is interpreted as a vector of asset returns, and Q represents
the covariance between assets, which forms a penalty on overall
covariance (risk) weighted by lambda.
"""

import cvxpy as cp
import numpy as np
import time
import torch
import torch.nn as nn
import neuromancer.slim as slim
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects

from neuromancer.trainer import Trainer
from neuromancer.problem import Problem
from neuromancer.constraint import variable
from neuromancer.dataset import DictDataset
from neuromancer.loss import PenaltyLoss
from neuromancer.modules import blocks
from neuromancer.system import Node

from portfolio_utils import gen_portfolio_lto_data, cvx_qp
import neuromancer_utils
import cvxpy as cp
from cvxpylayers.torch.cvxpylayer import CvxpyLayer

import argparse

import pandas as pd
import portfolio_utils

from metric_optimization import optimal_metric_QP

parser = argparse.ArgumentParser()
"""
n_samples = 1500
batch = 100
lrn_rate = 0.01
epochs = 1500
n_opt_steps = 150
train_p = False
learn_P=False
pred_P=True
hotstart=False
hotstart_layers_base = 2
hotstart_layers_each = 1
metric_layers_base = 2
metric_layers_each = 1
"""

parser.add_argument('--n_train',          type=int,   default=1000)
parser.add_argument('--n_valid',          type=int,   default=100)
parser.add_argument('--n_test',           type=int,   default=100)
parser.add_argument('--n_dim',            type=int,   default=250)
parser.add_argument('--batch_size',       type=int,   default=50)
parser.add_argument('--DR_lr',            type=float, default=1e-2)
parser.add_argument('--pretrain_epochs',  type=int,   default=150)
parser.add_argument('--DR_epochs',        type=int,   default=75)
parser.add_argument('--initial_steps',    type=int,   default=0)
parser.add_argument('--final_steps',      type=int,   default=0)
parser.add_argument('--n_opt_steps',      type=int,   default=50)
parser.add_argument('--initial_lb_P',     type=float, default=1000.0)
parser.add_argument('--initial_ub_P',     type=float, default=(1.0/1000.0))
parser.add_argument('--lb_P',             type=float, default=1.0)
parser.add_argument('--ub_P',             type=float, default=1.0)
parser.add_argument('--slack_mode',       type=int,   default=0)
parser.add_argument('--project_fixedpt',  type=int,   default=0)
parser.add_argument('--double_precision', type=int,   default=1)
parser.add_argument('--theory_optimal',   type=int,   default=0)
parser.add_argument('--regression_loss',  type=int,   default=0)
parser.add_argument('--seed',             type=int,   default=1)
parser.add_argument('--index',            type=int,   default=1)
args = parser.parse_args()


print("\n\n===========================================================")
print("Metric Learning for Douglas-Rachford Quadratic Programming")
print("Portfolio Optimization Problem")
print("Settings:")
for k,v in vars(args).items():
    spaces = " "*(30-len(k))
    print("{}:{}{}".format(k,spaces,v))
print("===========================================================\n\n")

"""
# # #  Dataset
"""
data_seed = args.seed
np.random.seed(data_seed)
torch.manual_seed(args.seed)

batsize = args.batch_size
n_dim   = args.n_dim
n_train = args.n_train
n_valid = args.n_valid
n_test  = args.n_test

initial_steps = args.initial_steps
initial_lb_P = args.initial_lb_P
initial_ub_P = args.initial_ub_P
lb_P = args.lb_P
ub_P = args.ub_P
project_fixedpt = args.project_fixedpt
precision = torch.float64 if args.double_precision else torch.float32



#batsize = 3
#n_dim = 5
#n_train = 10
#n_valid = 5
#n_test = 5
#
#DR_train_epochs = 1
#DR_steps = 500
#
#initial_steps = 1
#initial_lb_P = 1e-8
#initial_ub_P = 1e8
#project_fixedpt=False


#nsim = 100  # number of datapoints: increase sample density for more robust results

# create dictionaries with sampled datapoints with uniform distribution
#data_loaded = np.load('portfolio_data/portfolio_var50_ineq50_eq1_ex12000.npz', allow_pickle=True)
data_loaded = gen_portfolio_lto_data(n_dim,n_train,n_valid,n_test)
Q_load = data_loaded['Q']
A_load = data_loaded['A']
G_load = data_loaded['G']
h_load = data_loaded['h']
x_load = data_loaded['x']
p_train = data_loaded['trainX']
p_valid = data_loaded['validX']
p_test  = data_loaded['testX']
osqp_train = data_loaded['trainY']
osqp_train = torch.Tensor(osqp_train)
osqp_valid = data_loaded['validY']
osqp_valid = torch.Tensor(osqp_valid)
osqp_test  = data_loaded['testY']
osqp_test  = torch.Tensor(osqp_test)
Q = torch.Tensor(Q_load).to(precision)  #   # torch.eye(Q_load.shape[0])
Q_cond = torch.linalg.cond(Q).item()

if args.theory_optimal:
    H = 2*Q
    A = torch.eye(Q.shape[0]).to(precision)
    print("Solving for optimal metric")
    P_opt, gamma_opt = optimal_metric_QP(H,A)
    P_opt = P_opt.to(precision)
    print("Done")

print("Condition number of Q:")
print( Q_cond )



samples_train = {"p": torch.Tensor(p_train), "x_opt": osqp_train}  # JK TODO fix this, reduced size for debugging
samples_dev   = {"p": torch.Tensor(p_valid), "x_opt": osqp_valid}
samples_test  = {"p": torch.Tensor(p_test ), "x_opt": osqp_test}


#cvxpy_layer = cvx_qp(n_dim,Q)
#x_cvxpy_test = cvxpy_layer(samples_test['p'])
#test_Y = portfolio_utils.solve_convexqp(Q_load, samples_test['p'].T.numpy(), A_load, G_load, h_load, x_load)



# create named dictionary datasets
train_data = DictDataset(samples_train, name='train')
dev_data   = DictDataset(samples_dev,   name='dev')
test_data  = DictDataset(samples_test,  name='test')
# create torch dataloaders for the Trainer
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batsize, num_workers=0,
                                            collate_fn=train_data.collate_fn, shuffle=True)
dev_loader   = torch.utils.data.DataLoader(dev_data, batch_size=batsize, num_workers=0,
                                            collate_fn=dev_data.collate_fn, shuffle=True)
test_loader  = torch.utils.data.DataLoader(test_data, batch_size=batsize, num_workers=0,
                                            collate_fn=test_data.collate_fn, shuffle=True)
# note: training quality will depend on the DataLoader parameters such as batch size and shuffle


""

"""
# # #  pQP primal solution map architecture
"""
# define neural architecture for the solution map
func = blocks.MLP(insize=n_dim, outsize=n_dim,
                bias=True,
                linear_map=slim.maps['linear'],
                nonlin=nn.ReLU,
                hsizes=[n_dim*2] * 4)
# define symbolic solution map with concatenated features (problem parameters)
#xi = lambda p1, p2: torch.cat([p1, p2], dim=-1)
#features = Node(xi, ['p1', 'p2'], ['xi'], name='features')
sol_map = Node(func, ['p'], ['x'], name='map')
# trainable components of the problem solution
components = [sol_map]




"""
# # # objective and constraints formulation in Neuromancer
"""
# variables
x = variable("x")

# sampled parameters
p = variable('p')

# objective function
lambd = 1.0
f = torch.sum(p*x, dim = 1) + torch.sum( x*torch.matmul(Q.float(),x.T).T, dim=1 ) #-p@x + lambd * x@Q@x
obj = f.minimize(weight=1.0, name='obj')
objectives = [obj]

# constraints
e = torch.ones(n_dim)
Q_con = 100.
con_1 = Q_con*(torch.sum(x, dim=1) == 1) #Q_con*(e@x == 1)
con_1.name = 'c1'
con_2 = Q_con * (x >= 0)
con_2.name = 'c2'

constraints = [con_1, con_2]






"""
# # #  problem formulation in Neuromancer
"""
# create penalty method loss function
loss = PenaltyLoss(objectives, constraints)
# construct constrained optimization problem
problem = Problem(components, loss)


"""
# # #  problem solution in Neuromancer
"""
optimizer = torch.optim.AdamW(problem.parameters(), lr=1e-3)
# define trainer
trainer = Trainer(
    problem,
    train_loader,
    dev_loader,
    test_loader,
    optimizer,
    epochs=args.pretrain_epochs,
    patience=20,
    warmup=50,
    train_metric="train_loss",
    dev_metric="dev_loss",
    test_metric="test_loss",
    eval_metric="dev_loss",
)

# Train solution map
best_model = trainer.train()

with torch.no_grad():
    samples_test['name'] = 'test'
    prediction_model_out = problem(samples_test)

x_nm_test_prediction       = prediction_model_out['test_' + "x"].detach()

#error_L1_test_prediction = torch.nn.L1Loss()(x_nm_test_prediction,osqp_test).item()
error_L2_test_prediction = torch.norm(x_nm_test_prediction-osqp_test,p=2,dim=1).mean().item()


print("\nNeuromancer prediction L2 error: {}\n".format(error_L2_test_prediction))


'''
#######################################
#######################################

DR correction layers: Baseline and Training

#######################################
########################################
'''


import DRSolver as dr
from importlib import reload

reload(dr)


'''
# OBJECTIVE
'''
# x is assumed to include slack variables!
def f_obj(x,p):
    x = x[:n_dim]
    p = p[:n_dim]
    return p@x + lambd * x@(Q@x)            #lambd * torch.sum(x*torch.mv(Q,x))   #torch.sum( -p*x  ) +          #-p@x + lambd * x@(Q@x)



'''
# CONSTRAINTS
'''
def F_eq(x,p):
    #x = x[:n_dim]   # careful: n_dim becomes x_dim + n_ineq and x_dim becomes n_dim
    return (x.sum() - 1.0).unsqueeze(0)

'''
# SOLVER CLASS PARAMETERS
'''
x_dim = n_dim # dimension of primal variable
n_eq = 1
parm_dim = n_dim #number of problem parameters

'''
# TEST A CLASSICAL DR SOLVER WITH EUCLIDEAN METRIC
# no metric learning, initial steps or fixed-pt projection
'''

base_solver = dr.DRSolverQP(f_obj = f_obj,
                            F_eq = F_eq,
                            #F_ineq = F_ineq,
                            x_dim = x_dim,
                            n_eq = n_eq,
                            #n_ineq = 0,
                            #order = order,
                            JF_fixed=True,
                            parm_dim = parm_dim,
                            gamma = 2.0,
                            num_steps = args.n_opt_steps+args.initial_steps+args.final_steps ,
                            project_fixedpt = False,
                            slack_mode = args.slack_mode,
                            initial_steps= 0,
                            final_steps = 0,
                            initial_lb_P = 1.0,
                            initial_ub_P = 1.0,
                            lb_P = 1.0,
                            ub_P = 1.0,
                            precision = precision             )

# REMAP THROUGH DR CORRECTION
base_sol_map = Node(func, ['p'], ['x_predicted'], name='map')
base_DR_correction = Node(base_solver,['x_predicted','p'],['x','cnv_gap','cnv_gap_residuals'])
components = [base_sol_map, base_DR_correction]

### ADD A CONVERGENCE PENALTY
cnv_gap = variable("cnv_gap")
xtarget  = variable("x_opt")
x = variable("x")
f_cnv = (cnv_gap)**2 # (x - xtarget) #if args.regression_loss else (cnv_gap)**2  JK TODO: fix
obj = f_cnv.minimize(weight=1, name='cnv_obj')
objectives = [obj]
constraints = []

# create loss function
loss = PenaltyLoss(objectives, constraints)
problem = Problem(components, loss)

with torch.no_grad():
    samples_test['name'] = 'test'
    base_model_out = problem(samples_test)

x_nm_test_base       = base_model_out['test_' + "x"].detach()
cnv_gap_nm_test_base = base_model_out['test_' + "cnv_gap"].detach()
cnv_gap_residuals_nm_test_base = base_model_out['test_' + "cnv_gap_residuals"].mean(0).detach().tolist()

#resid_L1_test_base = torch.nn.L1Loss()(cnv_gap_nm_test_base,torch.zeros(cnv_gap_nm_test_base.shape)).item()
#error_L1_test_base = torch.nn.L1Loss()(x_nm_test_base,osqp_test).item()

resid_L2_test_base = torch.norm(cnv_gap_nm_test_base,    p=2,dim=1).mean().item()
error_L2_test_base = torch.norm(x_nm_test_base-osqp_test,p=2,dim=1).mean().item()


print("\nCorrection with euclidean DR baseline solver with {} steps:".format(args.n_opt_steps))
print("L2 error in fixed-point residual: {}".format(resid_L2_test_base))
print("L2 error from OSQP solution: {}\n".format(error_L2_test_base))

#input()

'''
# TRAIN A DR SOLVER WITH METRIC LEARNING
'''
gamma_in = gamma_opt if args.theory_optimal else 2.0
fixed_metric_in = P_opt if args.theory_optimal else None
solver = dr.DRSolverQP( f_obj = f_obj,
                        F_eq = F_eq,
                        #F_ineq = F_ineq,
                        x_dim = x_dim,
                        n_eq = n_eq,
                        #n_ineq = 0,
                        #order = order,
                        JF_fixed=False,
                        parm_dim = parm_dim,
                        num_steps = args.n_opt_steps,
                        project_fixedpt = args.project_fixedpt,
                        slack_mode = args.slack_mode,
                        initial_steps= args.initial_steps,
                        final_steps = args.final_steps,
                        initial_lb_P = args.initial_lb_P,
                        initial_ub_P = args.initial_ub_P,
                        lb_P = args.lb_P,
                        ub_P = args.ub_P,
                        precision = precision,
                        gamma = gamma_in,
                        fixed_metric = fixed_metric_in      )





# REMAP THROUGH DR CORRECTION
sol_map = Node(func, ['p'], ['x_predicted'], name='map')
DR_correction = Node(solver,['x_predicted','p'],['x','cnv_gap','cnv_gap_residuals'])
components = [sol_map, DR_correction]



### ADD A CONVERGENCE PENALTY
cnv_gap = variable("cnv_gap")
f_cnv = (x - xtarget)**2 #if args.regression_loss else (cnv_gap)**2  JK TODO: fix
obj = f_cnv.minimize(weight=1e8, name='cnv_obj')
objectives = [obj]
constraints = []



# create loss function
loss = PenaltyLoss(objectives, constraints)
# construct constrained optimization problem
problem = Problem(components, loss)




'''
#######################################
#######################################

Test DR layer before training

#######################################
########################################
'''

with torch.no_grad():
    t = time.time()
    samples_test['name'] = 'test'
    model_out = problem(samples_test)
    nm_time = time.time() - t

x_nm_test_pre = model_out['test_' + "x"].detach()
cnv_gap_nm_test_pre = model_out['test_' + "cnv_gap"].detach()
cnv_gap_residuals_nm_test_pre = model_out['test_' + "cnv_gap_residuals"].mean(0).detach().tolist()


resid_L2_test_pretrain = torch.norm(cnv_gap_nm_test_pre,p=2,dim=1).mean().item()
error_L2_test_pretrain = torch.norm(x_nm_test_pre-osqp_test,p=2,dim=1).mean().item()



print("\nCorrection with learnable DR solver with {} steps  BEFORE training:".format(args.n_opt_steps))
print("L2 error in fixed-point residual: {}".format(resid_L2_test_pretrain))
print("L2 error from OSQP solution: {}\n".format(error_L2_test_pretrain))


#input()

'''
#######################################
#######################################

Train DR correction layer

#######################################
########################################
'''

optimizer = torch.optim.AdamW(solver.parameters(), lr=args.DR_lr)
# define trainer
#logger = neuromancer_utils.LoggerChild()
callback = neuromancer_utils.CallbackChild()
trainer = Trainer(
    problem,
    train_loader,
    dev_loader,
    test_loader,
    optimizer,
    #logger=logger,
    callback=callback,
    epochs=args.DR_epochs,
    patience=15,
    warmup=10,
    train_metric="train_loss",
    dev_metric="dev_loss",
    test_metric="test_loss",
    eval_metric="dev_loss",
)
# Train solution map
best_model = trainer.train()

with torch.no_grad():
    t = time.time()
    samples_test['name'] = 'test'
    model_out = problem(samples_test)
    nm_time = time.time() - t


x_nm_test_post = model_out['test_' + "x"].detach()
cnv_gap_nm_test_post = model_out['test_' + "cnv_gap"].detach()
cnv_gap_residuals_nm_test_post = model_out['test_' + "cnv_gap_residuals"].mean(0).detach().tolist()


resid_L2_test_posttrain = torch.norm(cnv_gap_nm_test_post,p=2,dim=1).mean().item()
error_L2_test_posttrain = torch.norm(x_nm_test_post-osqp_test,p=2,dim=1).mean().item()


print("\nCorrection with learnable DR solver with {} steps AFTER training:".format(args.n_opt_steps))
print("L2 error in fixed-point residual: {}".format(resid_L2_test_posttrain))
print("L2 error from OSQP solution: {}\n".format(error_L2_test_posttrain))


#check for [tensor]

csv_outs = vars(args)
csv_outs['Q_cond'] = Q_cond
csv_outs['resid_L2_test_base']       = resid_L2_test_base
csv_outs['resid_L2_test_pretrain']   = resid_L2_test_pretrain
csv_outs['resid_L2_test_posttrain']  = resid_L2_test_posttrain

csv_outs['error_L2_test_prediction'] = error_L2_test_prediction
csv_outs['error_L2_test_base']       = error_L2_test_base
csv_outs['error_L2_test_pretrain']   = error_L2_test_pretrain
csv_outs['error_L2_test_posttrain']  = error_L2_test_posttrain


for i in range(len(cnv_gap_residuals_nm_test_base)):
    csv_outs["cnv_gap_residuals_base_{}".format(i)] = cnv_gap_residuals_nm_test_base[i]

for i in range(len(cnv_gap_residuals_nm_test_pre)):
    csv_outs["cnv_gap_residuals_pre_{}".format(i)] = cnv_gap_residuals_nm_test_pre[i]

for i in range(len(cnv_gap_residuals_nm_test_post)):
    csv_outs["cnv_gap_residuals_post_{}".format(i)] = cnv_gap_residuals_nm_test_post[i]



for i in range(len(trainer.train_losses_epoch)):
    csv_outs["train_losses_epoch_{}".format(i)] = trainer.train_losses_epoch[i]

for i in range(len(trainer.dev_losses_epoch)):
    csv_outs["dev_losses_epoch_{}".format(i)] = trainer.dev_losses_epoch[i]

for i in range(len(trainer.resid_L2_dev_epoch)):
    csv_outs["dev_losses_epoch_{}".format(i)] = trainer.resid_L2_dev_epoch[i]




csv_outs = {k:[v] for (k,v) in csv_outs.items()}


print("csv_outs")
print( csv_outs )

df_outs = pd.DataFrame.from_dict(csv_outs)
outPathCsv = './csv/'+ "portfolioQP" + "_" + str(args.index) + ".csv"
df_outs.to_csv(outPathCsv)

print("Saved csv output to {}.".format(outPathCsv))
