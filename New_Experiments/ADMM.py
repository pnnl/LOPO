


'''
This script implements ADMM

following the papers

https://web.stanford.edu/~boyd/papers/pdf/metric_select_DR_ADMM.pdf
https://web.stanford.edu/~boyd/papers/pdf/metric_select_fdfbs.pdf


'''


import torch
import torch.nn as nn
import numpy as np
import Prox as px

import Metric as mx
from torch.func import vmap
from importlib import reload
reload(px)
reload(mx)



class ADMMSolver(nn.Module):
    """
    Implementation of an ADMM Solution routine for corrections of solution estimates for problems of the form
    
    min f(x)
    subject to:
    F_ineq(x) <= 0
    F_eq(x)= 0
    
    The problem is reformulated as
    
    min f(x)
    subject to:
    F(x,s) = 0
    s>=0
    
    for slack variables s, and F(x,s) defined as

    F(x,s) = [ F_eq(x) ; F_ineq(x) + s ]

    """
    def __init__(self,f_obj = None,
                 F_ineq = None,
                 F_eq = None,
                 x_dim = 0,
                 n_ineq = 0,
                 n_eq = 0,
                 JF_fixed = False,
                 parm_dim = None,
                 num_steps=3,
                 Metric = None,
                 gamma = 1.0):
        """
        :param f_obj: functorch compatible function) a parameterized function f with input of the form (x,parms) where parms is a tensor that matches the batch of x, and has last dim parm features.
                  f is defined unbatched, the method will call vmap, to "raise" f to batch dim
                  gives the objective to be optimized
        :param F_ineq:(functorch compatible function) a parameterized function F with input of the form (x, parms) where parms is a tensor that matches the batch of x, and has last dim parm features.
                  F is defined unbatched, the method will call vmap, to "raise" f to batch dim
                  gives the inequality constraints to satisfy, F_ineq(x) <= 0
        :param F_eq:(functorch compatible function) a parameterized function F with input of the form (x, parms) where parms is a tensor that matches the batch of x, and has last dim parm features.
                  F is defined unbatched, the method will call vmap, to "raise" f to batch dim
                  gives the equality constraints to satisfy, F_eq(x) = 0
        :param x_dim: (int) dimension of the primal variables
        :param n_ineq: (int) number of inequality constraints
        :param n_eq: (int) number of equality constraints
        :param order: (str) one of {'first','second'} the order of the approximation used for f_obj
        :param JF_fixed: (Bool) Indicates if the Jacobian of F should be computed at each iteration. Default is False, if True Jacobian of F will be precomputed at x=0, parms = 0
        :param parm_dim: (int) the dimension of parms for precomputing Jacobian
        :param num_steps: (int) number of iteration steps for the Douglas Rachford method using Identity metric
        :param initial_steps: (int) number of learned metric DR steps to take, default is one.
        """
        super().__init__()
        self.x_dim = x_dim
        self.n_ineq = n_ineq
        self.n_eq = n_eq
        self.num_steps = num_steps
        self.JF_fixed = JF_fixed
        self.parm_dim = parm_dim
        self.n_dim = self.x_dim + self.n_ineq
        self.f_obj = f_obj
        self.F_ineq = F_ineq
        self.F_eq = F_eq
        self.gamma = gamma
        self.alpha = 0.5
        if n_eq > x_dim: print('ERROR: Equality constraints are overdetermined')
        #### Convert problem inputs to the standard form for the DR iterations
        #i.d. problem type
        #pid =
        #    = 1 only equality constraints
        #    = 2 only inequality constraints
        #    = 3 both equality and inequality constraints
        #    = 0 Error: no constraints
        pid = 2*(self.F_ineq != None) + (self.F_eq != None)
        self.pid = pid
        if pid == 0: print( 'ERROR: One of F_eq or F_ineq must be defined')
        if pid == 1:
            def f(xs, parms):
                return self.f_obj(xs,parms)
            def F(xs, parms):
                return self.F_eq(xs,parms)
        if pid == 2:
            def f(xs,parms):
                x = xs[0:self.x_dim]
                s = xs[self.x_dim:]
                return self.f_obj(x,parms)
            def F(xs,parms):
                x = xs[0:self.x_dim]
                s = xs[self.x_dim:]
                return self.F_ineq(x,parms) + s
        if pid ==3 :
            def Fs_ineq(xs,parms):
                x = xs[0:self.x_dim]
                s = xs[self.x_dim:]
                return torch.cat( (torch.zeros(self.n_eq),  self.F_ineq(x,parms) + s))
            def Fs_eq(xs,parms):
                x = xs[0:self.x_dim]
                return torch.cat( (self.F_eq(x,parms), torch.zeros(self.n_ineq) ))
            def F(xs, parms):
                return Fs_eq(xs,parms) + Fs_ineq(xs,parms)
            def f(xs,parms):
                x = xs[0:self.x_dim]
                s = xs[self.x_dim:]
                return self.f_obj(x,parms)
        self.F = F
        self.f = f
        if Metric!= None:
            self.Pm = Metric
        else:
            self.Pm = Identity(self.n_dim,self.parm_dim)
        #adjust gamma to match with ADMM computation
        self.gamma = 1/(2*self.gamma)
        self.foF_Id = px.SecondOrderObjectiveConstraintComposition(self.f,self.F,self.Pm,JF_fixed = self.JF_fixed,n_dim = self.n_dim, parm_dim = self.parm_dim, gamma = self.gamma)
        
        ## define the slack bounds
        upper_bound = 1e3*torch.ones(self.n_dim)
        if self.n_ineq>0:
            lower_bound = torch.cat( (-1e3*torch.ones(self.x_dim),torch.zeros(self.n_ineq)))
        else:
            lower_bound = torch.zeros(self.n_dim)
        def f_upper(parms):
            return upper_bound
        def f_lower(parms):
            return lower_bound
        self.sp = px.BoxConstraint(f_lower,f_upper)

    def forward(self,x,parms):
        if self.n_ineq>0 :
            #add the slack variables
            x = torch.cat((x,torch.zeros((x.shape[0],self.n_ineq))),dim = -1)
        y_k = x
        w_k = torch.zeros(x.shape)
        x_hist = []
        x_hist.append(x)
        for n in range(self.num_steps):
            y_k_old = y_k
            x_k = self.foF_Id(y_k - w_k,parms)
            xA_k = 2*self.alpha*x_k + (1-2*self.alpha)*y_k
            y_k = self.sp(xA_k + w_k,parms)
            w_k = w_k + xA_k - y_k
            r_gap = x_k - y_k
            s_gap = y_k - y_k_old
            x_hist.append(x_k)
        if self.n_ineq> 0:
            x_out =  x_k[:,:-self.n_ineq]   # JK bug note: this indexing causes return
        if self.n_ineq == 0:
            x_out =  x_k
        return x_out, r_gap, s_gap, x_hist





















class ADMMSolverFast(nn.Module):
    """
    Implementation of an ADMM Solution routine for corrections of solution estimates for problems of the form
    
    min f(x)
    subject to:
    F_ineq(x) <= 0
    F_eq(x)= 0
    
    The problem is reformulated as
    
    min f(x)
    subject to:
    F(x,s) = 0
    s>=0
    
    for slack variables s, and F(x,s) defined as

    F(x,s) = [ F_eq(x) ; F_ineq(x) + s ]

    """
    def __init__(self,f_obj = None,
                 F_ineq = None,
                 F_eq = None,
                 x_dim = 0,
                 n_ineq = 0,
                 n_eq = 0,
                 JF_fixed = False,
                 parm_dim = None,
                 num_steps=3,
                 Metric = None,
                 gamma = 1.0):
        """
        :param f_obj: functorch compatible function) a parameterized function f with input of the form (x,parms) where parms is a tensor that matches the batch of x, and has last dim parm features.
                  f is defined unbatched, the method will call vmap, to "raise" f to batch dim
                  gives the objective to be optimized
        :param F_ineq:(functorch compatible function) a parameterized function F with input of the form (x, parms) where parms is a tensor that matches the batch of x, and has last dim parm features.
                  F is defined unbatched, the method will call vmap, to "raise" f to batch dim
                  gives the inequality constraints to satisfy, F_ineq(x) <= 0
        :param F_eq:(functorch compatible function) a parameterized function F with input of the form (x, parms) where parms is a tensor that matches the batch of x, and has last dim parm features.
                  F is defined unbatched, the method will call vmap, to "raise" f to batch dim
                  gives the equality constraints to satisfy, F_eq(x) = 0
        :param x_dim: (int) dimension of the primal variables
        :param n_ineq: (int) number of inequality constraints
        :param n_eq: (int) number of equality constraints
        :param order: (str) one of {'first','second'} the order of the approximation used for f_obj
        :param JF_fixed: (Bool) Indicates if the Jacobian of F should be computed at each iteration. Default is False, if True Jacobian of F will be precomputed at x=0, parms = 0
        :param parm_dim: (int) the dimension of parms for precomputing Jacobian
        :param num_steps: (int) number of iteration steps for the Douglas Rachford method using Identity metric
        :param initial_steps: (int) number of learned metric DR steps to take, default is one.
        """
        super().__init__()
        self.x_dim = x_dim
        self.n_ineq = n_ineq
        self.n_eq = n_eq
        self.num_steps = num_steps
        self.JF_fixed = JF_fixed
        self.parm_dim = parm_dim
        self.n_dim = self.x_dim + self.n_ineq
        self.f_obj = f_obj
        self.F_ineq = F_ineq
        self.F_eq = F_eq
        self.gamma = gamma
        self.alpha = 0.5
        if n_eq > x_dim: print('ERROR: Equality constraints are overdetermined')
        #### Convert problem inputs to the standard form for the DR iterations
        #i.d. problem type
        #pid =
        #    = 1 only equality constraints
        #    = 2 only inequality constraints
        #    = 3 both equality and inequality constraints
        #    = 0 Error: no constraints
        pid = 2*(self.F_ineq != None) + (self.F_eq != None)
        self.pid = pid
        if pid == 0: print( 'ERROR: One of F_eq or F_ineq must be defined')
        if pid == 1:
            def f(xs, parms):
                return self.f_obj(xs,parms)
            def F(xs, parms):
                return self.F_eq(xs,parms)
        if pid == 2:
            def f(xs,parms):
                x = xs[0:self.x_dim]
                s = xs[self.x_dim:]
                return self.f_obj(x,parms)
            def F(xs,parms):
                x = xs[0:self.x_dim]
                s = xs[self.x_dim:]
                return self.F_ineq(x,parms) + s
        if pid ==3 :
            def Fs_ineq(xs,parms):
                x = xs[0:self.x_dim]
                s = xs[self.x_dim:]
                return torch.cat( (torch.zeros(self.n_eq),  self.F_ineq(x,parms) + s))
            def Fs_eq(xs,parms):
                x = xs[0:self.x_dim]
                return torch.cat( (self.F_eq(x,parms), torch.zeros(self.n_ineq) ))
            def F(xs, parms):
                return Fs_eq(xs,parms) + Fs_ineq(xs,parms)
            def f(xs,parms):
                x = xs[0:self.x_dim]
                s = xs[self.x_dim:]
                return self.f_obj(x,parms)
        self.F = F
        self.f = f
        if Metric!= None:
            self.Pm = Metric
        else:
            self.Pm = Identity(self.n_dim,self.parm_dim)
        #adjust gamma to match with ADMM computation
        self.gamma = 1/(2*self.gamma)
        self.foF_Id = px.SecondOrderObjectiveConstraintCompositionFixedH(self.f,self.F,self.Pm,JF_fixed = self.JF_fixed,n_dim = self.n_dim, parm_dim = self.parm_dim, gamma = self.gamma)
        
        ## define the slack bounds
        upper_bound = 1e3*torch.ones(self.n_dim)
        if self.n_ineq>0:
            lower_bound = torch.cat( (-1e3*torch.ones(self.x_dim),torch.zeros(self.n_ineq)))
        else:
            lower_bound = torch.zeros(self.n_dim)
        def f_upper(parms):
            return upper_bound
        def f_lower(parms):
            return lower_bound
        self.sp = px.BoxConstraint(f_lower,f_upper)

    def forward(self,x,parms):
        if self.n_ineq>0 :
            #add the slack variables
            x = torch.cat((x,torch.zeros((x.shape[0],self.n_ineq))),dim = -1)
        y_k = x
        w_k = torch.zeros(x.shape)
        x_hist = []
        x_hist.append(x)
        Lcho, Md = self.foF_Id.prefactor(x,parms)
        for n in range(self.num_steps):
            y_k_old = y_k
            x_k = self.foF_Id(y_k - w_k,parms,Lcho,Md)
            xA_k = 2*self.alpha*x_k + (1-2*self.alpha)*y_k
            y_k = self.sp(xA_k + w_k,parms)
            w_k = w_k + xA_k - y_k
            r_gap = x_k - y_k
            s_gap = y_k - y_k_old
            x_hist.append(x_k)
        if self.n_ineq> 0:
            x_out =  x_k[:,:-self.n_ineq]   # JK bug note: this indexing causes return
        if self.n_ineq == 0:
            x_out =  x_k
        return x_out, r_gap, s_gap, x_hist

    



























'''
Define Metric Functions that allow for 
solving 'preconditioned' version of ADMM

Identity is the base class, it allows for solution of standard ADMM

OptMetric is a class that allows for specifying a metric, that will be applied like a preconditioning matrix on the algorithm following

https://web.stanford.edu/~boyd/papers/pdf/metric_select_DR_ADMM.pdf

Though this metric choice must currently be computed by the user.
'''


class Identity(torch.nn.Module):
    def __init__(self,n_dim,parm_dim):
        super().__init__()
        self.n_dim = n_dim
        self.P_d = torch.nn.Parameter(torch.ones(self.n_dim),requires_grad = False)
    def forward(self,x,parms):
        Pm = torch.diag(self.P_d)
        return Pm
    def return_E(self,x,parms):
        Pm = torch.diag(self.P_d)
        return Pm
    


class OptMetric(torch.nn.Module):
    def __init__(self,n_dim,parm_dim,E):
        super().__init__()
        self.E = torch.nn.Parameter(torch.tensor(E,dtype = torch.float32),requires_grad = False)
        self.M = torch.nn.Parameter(torch.tensor(np.matmul(np.transpose(E),E),dtype = torch.float32 ),requires_grad = False)
    def forward(self,x,parms):
        return self.M
    def return_E(self,x,parms):
        return self.E
    



class ParametricDiagonal(torch.nn.Module):
    def __init__(self,n_dim,parm_dim,upper_bound,lower_bound,scl_upper_bound = 0.2,scl_lower_bound = 0.05):
        super().__init__()
        self.n_dim = n_dim
        self.P_diag_upper_bound = upper_bound
        self.P_diag_lower_bound = lower_bound
        self.parm_dim = parm_dim
        #self.hidden_dim = 10
        self.hidden_dim = np.round(10*self.parm_dim).astype(int)
        #self.hidden_dim = np.round(2*self.parm_dim).astype(int)
        self.DiagMap = nn.Sequential(
          nn.Linear(self.parm_dim,self.hidden_dim),
          nn.ReLU(),
          nn.Linear(self.hidden_dim,self.hidden_dim),
          nn.ReLU(),
          nn.Linear(self.hidden_dim,self.n_dim),
        )
        self.scl_upper_bound = scl_upper_bound
        self.scl_lower_bound = scl_lower_bound
        self.ScaleMap = nn.Sequential(
                nn.Linear(self.parm_dim,self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim,self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim,1)
                )
    def forward(self,x,parms):
        Pd = self.DiagMap(parms)
        scl = self.scl_lower_bound +  torch.sigmoid(self.ScaleMap(parms))*( self.scl_upper_bound - self.scl_lower_bound)
        P_diag = scl*( self.P_diag_lower_bound + torch.sigmoid(Pd)*(self.P_diag_upper_bound - self.P_diag_lower_bound) )
        Pm = torch.diag(P_diag)
        return Pm
    def scl_comp(self,x,parms):
        scl = torch.sigmoid(self.ScaleMap(parms))
        return scl

