
import torch
import torch.nn as nn

import Prox as px

import Metric as mx
from torch.func import vmap
from importlib import reload
reload(px)
reload(mx)

import time



class DRSolver(nn.Module):
    """
    Implementation of a Parameteric Douglas Rachford (DR) Solution routine for corrections of solution estimates for problems of the form
    
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

    DR is an operator splitting approach, here applied to the splitting

    min g_1(x,s) + g_2(x,s)

    with
     g_1(x,s) = f(x) + i_{ (x,s) : F(x,s) = 0}
     g_2(x) = i_{ s : s>=0 }

    where i_{S} is the indicator function on set S.

    This Routine uses a two stage approach

    """
    def __init__(self,f_obj = None,
                 F_ineq = None,
                 F_eq = None,
                 x_dim = 0,
                 n_ineq = 0,
                 n_eq = 0,
                 order = 'first',
                 JF_fixed = False,
                 parm_dim = None,
                 num_steps=3,
                 initial_steps = 1,
                 initial_lb_P = 1e-8,
                 initial_ub_P = 1e8,
                 lb_P = 1.0/2.0,
                 ub_P = 2.0,
                 scl_lb_P = 0.05,
                 scl_ub_P = 0.2,
                 project_fixedpt=True):
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
        self.initial_steps = initial_steps
        self.order = order
        self.JF_fixed = JF_fixed
        self.parm_dim = parm_dim
        self.n_dim = self.x_dim + self.n_ineq
        self.f_obj = f_obj
        self.F_ineq = F_ineq
        self.F_eq = F_eq

        self.initial_lb_P = initial_lb_P
        self.initial_ub_P = initial_ub_P
        self.lb_P = lb_P
        self.ub_P = ub_P
        self.project_fixedpt = project_fixedpt

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
        #### Set the Prox of g_1(x,s)
        # JK TODO: if lb=ub, use Identity
        self.Pm = mx.ParametricDiagonal(self.n_dim,self.parm_dim,initial_ub_P,initial_lb_P,scl_upper_bound=scl_ub_P,scl_lower_bound=scl_lb_P)
        if self.order == 'first': self.foF = px.FirstOrderObjectiveConstraintComposition(self.f,self.F,self.Pm,JF_fixed = self.JF_fixed,n_dim = self.n_dim, parm_dim = self.parm_dim)
        if self.order == 'second': self.foF = px.SecondOrderObjectiveConstraintComposition(self.f,self.F,self.Pm,JF_fixed = self.JF_fixed,n_dim = self.n_dim, parm_dim = self.parm_dim)
        self.Pm_2 = mx.ParametricDiagonal(self.n_dim,self.parm_dim,ub_P,lb_P,scl_upper_bound=scl_ub_P,scl_lower_bound=scl_lb_P)
        #self.Pm_2 = mx.Identity(self.n_dim,self.parm_dim)
        #self.Pm_2 = mx.ParametricStateDiagonal(self.n_dim,self.parm_dim,ub_P,lb_P,scl_upper_bound=scl_ub_P,scl_lower_bound=scl_lb_P)
        #self.Pm_2 = mx.StaticDiagonal(n_dim = self.n_dim, parm_dim = self.parm_dim)
        if self.order == 'first':  self.foF_Id =  px.FirstOrderObjectiveConstraintComposition(self.f,self.F,self.Pm_2,JF_fixed = self.JF_fixed,n_dim = self.n_dim, parm_dim = self.parm_dim)
        if self.order == 'second': self.foF_Id = px.SecondOrderObjectiveConstraintComposition(self.f,self.F,self.Pm_2,JF_fixed = self.JF_fixed,n_dim = self.n_dim, parm_dim = self.parm_dim)
        
        ### Set the Prox of g_2(x,s)
        ## define the slack bounds
        upper_bound = 1e3*torch.ones(self.n_dim)
        lower_bound = torch.cat( (-1e3*torch.ones(self.x_dim),torch.zeros(self.n_ineq)))
        def f_upper(parms):
            return upper_bound
        def f_lower(parms):
            return lower_bound
        self.sp = px.BoxConstraint(f_lower,f_upper)
        #up_bound_fp = 1e8
        #low_bound_fp = 1e-8
        #self.fp_Pm = mx.ParametricStateDiagonal(self.n_dim,self.parm_dim,initial_ub_P,initial_lb_P)
        self.fp_Pm = mx.ParametricDiagonal(self.n_dim,self.parm_dim,initial_ub_P,initial_lb_P,scl_upper_bound=scl_ub_P,scl_lower_bound=scl_lb_P)

    def forward(self,x,parms):
        ## Use zero slack initialization
        if self.n_ineq > 0:
            #add the slack variables
            x = torch.cat((x,torch.zeros((x.shape[0],self.n_ineq))),dim = -1)
            #xz = torch.cat((x,torch.zeros((x.shape[0],self.n_ineq))),dim = -1)
            #xs = torch.vmap(self.F)(xz,parms)
            #slacks = -xs[:,-self.n_ineq:]
            #x = torch.cat((x,slacks),dim = -1)
            #if self.initial_steps>0:
            #     x = torch.cat((x,torch.zeros((x.shape[0],self.n_ineq))),dim = -1)
            #if self.initial_steps==0:
            #xz = torch.cat((x,torch.zeros((x.shape[0],self.n_ineq))),dim = -1)
            #xs = torch.vmap(self.F)(xz,parms)
            #slacks = -xs[:,-self.n_ineq:]
            #x = torch.cat((x,slacks),dim = -1)

        x_k = x
        #for n in range(self.initial_steps):
        #    y_k = self.sp(x_k,parms)
        #    z_k = self.foF(2*y_k - x_k,parms)
        #    x_k_new = x_k + (z_k - y_k)
        #    x_k = x_k_new

        #if self.project_fixedpt:
        #    x_k = self.FixedPointConditionProjection(x_k[:,:-self.n_ineq],parms)

        x_hist = []
        x_hist.append(x_k)
        for n in range(self.num_steps):
            y_k = self.sp(x_k,parms)
            z_k = self.foF_Id(2*y_k - x_k,parms)
            x_k_new = x_k + (z_k - y_k)
            x_k = x_k_new
            cnv_gap = z_k - y_k
            x_hist.append(x_k)
        scl_vals = vmap(self.Pm_2.scl_comp)(x,parms)
        return x_k_new[:,:-self.n_ineq], cnv_gap, x_hist   # JK bug note: this indexing causes return


    def FixedPointConditionProjection(self,x,parms):
        x_init = x
        #add initial slack variables
        xz = torch.cat((x,torch.zeros((x.shape[0],self.n_ineq))),dim = -1)
        xs = torch.vmap(self.F)(xz,parms)
        slacks = -xs[:,-self.n_ineq:]
        x = torch.cat((x,slacks),dim = -1)
        s_plus = torch.cat( (torch.zeros(x.shape[0],self.x_dim),torch.relu(slacks)),dim = -1)
        if self.order == 'second':
            batch_size = x.shape[0]
            Hx = self.foF.H_vec(x,parms)
            Id_batch = torch.tile(torch.unsqueeze(torch.eye(self.n_dim),dim=0),(batch_size,1,1))
            Md = (self.foF.gamma/2)*Id_batch + Hx
            grads = torch.linalg.solve(Md,self.foF.f_grad(x,parms))
        if self.order == 'first':
            grads = (self.foF.gamma/2)*self.foF.f_grad(x,parms)
        eta = s_plus + grads
        JFx = self.foF.JF(x,parms)
        ### Take a QR decomposition of the Jacobian
        with torch.no_grad():
            Q, R  = torch.linalg.qr(torch.transpose(JFx,1,2),mode = 'complete')
            null_dim = Q.shape[-1] - R.shape[-1]
            R = R[:,:-null_dim,:]
            Qr = Q[:,:,:-null_dim]
            Qn = Q[:,:,-null_dim:]
        P_mats = vmap(self.fp_Pm)(x,parms)
        QTPQm = torch.bmm(torch.transpose(Qr,1,2),torch.bmm(P_mats,Qr))
        # Compute the oblique projection
        xabs = torch.cat( (x_init,torch.abs(slacks)),dim = -1)
        xabs_vec = torch.unsqueeze(xabs,-1)
        eta_vec = torch.unsqueeze(eta,-1)
        z = torch.bmm(P_mats,xabs_vec - eta_vec)
        z = torch.bmm(torch.transpose(Qr,1,2),z)
        z = torch.linalg.solve(QTPQm,z)
        z = torch.bmm(Qr,z)
        z = z + eta_vec
        z = torch.squeeze(z,-1)
        new_slacks = z[:,-self.n_ineq:]
        new_slacks = 2*torch.relu(slacks) - new_slacks
        return torch.cat((z[:,0:-self.n_ineq],new_slacks),dim=-1)












class DRSolverFast(nn.Module):
    """
    Implementation of a Parameteric Douglas Rachford (DR) Solution routine for corrections of solution estimates for problems of the form
    
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

    DR is an operator splitting approach, here applied to the splitting

    min g_1(x,s) + g_2(x,s)

    with
     g_1(x,s) = f(x) + i_{ (x,s) : F(x,s) = 0}
     g_2(x) = i_{ s : s>=0 }

    where i_{S} is the indicator function on set S.

    This Routine uses a two stage approach

    """
    def __init__(self,f_obj = None,
                 F_ineq = None,
                 F_eq = None,
                 x_dim = 0,
                 n_ineq = 0,
                 n_eq = 0,
                 order = 'first',
                 JF_fixed = False,
                 parm_dim = None,
                 num_steps=3,
                 initial_steps = 1,
                 initial_lb_P = 1e-8,
                 initial_ub_P = 1e8,
                 lb_P = 1.0/2.0,
                 ub_P = 2.0,
                 scl_lb_P = 0.05,
                 scl_ub_P = 0.2,
                 project_fixedpt=True):
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
        self.initial_steps = initial_steps
        self.order = order
        self.JF_fixed = JF_fixed
        self.parm_dim = parm_dim
        self.n_dim = self.x_dim + self.n_ineq
        self.f_obj = f_obj
        self.F_ineq = F_ineq
        self.F_eq = F_eq

        self.initial_lb_P = initial_lb_P
        self.initial_ub_P = initial_ub_P
        self.lb_P = lb_P
        self.ub_P = ub_P
        self.project_fixedpt = project_fixedpt

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
        #### Set the Prox of g_1(x,s)
        # JK TODO: if lb=ub, use Identity
        self.Pm = mx.ParametricDiagonal(self.n_dim,self.parm_dim,initial_ub_P,initial_lb_P,scl_upper_bound=scl_ub_P,scl_lower_bound=scl_lb_P)
        if self.order == 'first': self.foF = px.FirstOrderObjectiveConstraintComposition(self.f,self.F,self.Pm,JF_fixed = self.JF_fixed,n_dim = self.n_dim, parm_dim = self.parm_dim)
        if self.order == 'second': self.foF = px.SecondOrderObjectiveConstraintComposition(self.f,self.F,self.Pm,JF_fixed = self.JF_fixed,n_dim = self.n_dim, parm_dim = self.parm_dim)
        self.Pm_2 = mx.ParametricDiagonal(self.n_dim,self.parm_dim,ub_P,lb_P,scl_upper_bound=scl_ub_P,scl_lower_bound=scl_lb_P)
        #self.Pm_2 = mx.Identity(self.n_dim,self.parm_dim)
        #self.Pm_2 = mx.ParametricStateDiagonal(self.n_dim,self.parm_dim,ub_P,lb_P,scl_upper_bound=scl_ub_P,scl_lower_bound=scl_lb_P)
        #self.Pm_2 = mx.StaticDiagonal(n_dim = self.n_dim, parm_dim = self.parm_dim)
        if self.order == 'first':  self.foF_Id =  px.FirstOrderObjectiveConstraintComposition(self.f,self.F,self.Pm_2,JF_fixed = self.JF_fixed,n_dim = self.n_dim, parm_dim = self.parm_dim)
        if self.order == 'second': self.foF_Id = px.SecondOrderObjectiveConstraintCompositionFixedH(self.f,self.F,self.Pm_2,JF_fixed = self.JF_fixed,n_dim = self.n_dim, parm_dim = self.parm_dim)
        
        ### Set the Prox of g_2(x,s)
        ## define the slack bounds
        upper_bound = 1e3*torch.ones(self.n_dim)
        lower_bound = torch.cat( (-1e3*torch.ones(self.x_dim),torch.zeros(self.n_ineq)))
        def f_upper(parms):
            return upper_bound
        def f_lower(parms):
            return lower_bound
        self.sp = px.BoxConstraint(f_lower,f_upper)
        #up_bound_fp = 1e8
        #low_bound_fp = 1e-8
        #self.fp_Pm = mx.ParametricStateDiagonal(self.n_dim,self.parm_dim,initial_ub_P,initial_lb_P)
        self.fp_Pm = mx.ParametricDiagonal(self.n_dim,self.parm_dim,initial_ub_P,initial_lb_P,scl_upper_bound=scl_ub_P,scl_lower_bound=scl_lb_P)

    def forward(self,x,parms):
        ## Use zero slack initialization
        if self.n_ineq > 0:
            #add the slack variables
            x = torch.cat((x,torch.zeros((x.shape[0],self.n_ineq))),dim = -1)
            #xz = torch.cat((x,torch.zeros((x.shape[0],self.n_ineq))),dim = -1)
            #xs = torch.vmap(self.F)(xz,parms)
            #slacks = -xs[:,-self.n_ineq:]
            #x = torch.cat((x,slacks),dim = -1)
            #if self.initial_steps>0:
            #     x = torch.cat((x,torch.zeros((x.shape[0],self.n_ineq))),dim = -1)
            #if self.initial_steps==0:
            #xz = torch.cat((x,torch.zeros((x.shape[0],self.n_ineq))),dim = -1)
            #xs = torch.vmap(self.F)(xz,parms)
            #slacks = -xs[:,-self.n_ineq:]
            #x = torch.cat((x,slacks),dim = -1)
        Ucho, Md = self.foF_Id.prefactor(x,parms)

        x_k = x
        #for n in range(self.initial_steps):
        #    y_k = self.sp(x_k,parms)
        #    z_k = self.foF(2*y_k - x_k,parms)
        #    x_k_new = x_k + (z_k - y_k)
        #    x_k = x_k_new

        #if self.project_fixedpt:
        #    x_k = self.FixedPointConditionProjection(x_k[:,:-self.n_ineq],parms)

        x_hist = []
        x_hist.append(x_k)
        cnv_gap_tot = 0
        for n in range(self.num_steps):
            y_k = self.sp(x_k,parms)
            z_k = self.foF_Id(2*y_k - x_k,parms,Ucho,Md)
            x_k_new = x_k + (z_k - y_k)
            x_k = x_k_new
            cnv_gap = z_k - y_k
            x_hist.append(x_k)
            #cnv_gap_tot = cnv_gap_tot + (n**3)*torch.sum(cnv_gap**2)
        scl_vals = vmap(self.Pm_2.scl_comp)(x,parms)
        cnv_gap = z_k - y_k
        return x_k_new, cnv_gap   # JK bug note: this indexing causes return
        #return x_k_new[:,:-self.n_ineq], cnv_gap, x_hist   # JK bug note: this indexing causes return


    def FixedPointConditionProjection(self,x,parms):
        x_init = x
        #add initial slack variables
        xz = torch.cat((x,torch.zeros((x.shape[0],self.n_ineq))),dim = -1)
        xs = torch.vmap(self.F)(xz,parms)
        slacks = -xs[:,-self.n_ineq:]
        x = torch.cat((x,slacks),dim = -1)
        s_plus = torch.cat( (torch.zeros(x.shape[0],self.x_dim),torch.relu(slacks)),dim = -1)
        if self.order == 'second':
            batch_size = x.shape[0]
            Hx = self.foF.H_vec(x,parms)
            Id_batch = torch.tile(torch.unsqueeze(torch.eye(self.n_dim),dim=0),(batch_size,1,1))
            Md = (self.foF.gamma/2)*Id_batch + Hx
            grads = torch.linalg.solve(Md,self.foF.f_grad(x,parms))
        if self.order == 'first':
            grads = (self.foF.gamma/2)*self.foF.f_grad(x,parms)
        eta = s_plus + grads
        JFx = self.foF.JF(x,parms)
        ### Take a QR decomposition of the Jacobian
        with torch.no_grad():
            Q, R  = torch.linalg.qr(torch.transpose(JFx,1,2),mode = 'complete')
            null_dim = Q.shape[-1] - R.shape[-1]
            R = R[:,:-null_dim,:]
            Qr = Q[:,:,:-null_dim]
            Qn = Q[:,:,-null_dim:]
        P_mats = vmap(self.fp_Pm)(x,parms)
        QTPQm = torch.bmm(torch.transpose(Qr,1,2),torch.bmm(P_mats,Qr))
        # Compute the oblique projection
        xabs = torch.cat( (x_init,torch.abs(slacks)),dim = -1)
        xabs_vec = torch.unsqueeze(xabs,-1)
        eta_vec = torch.unsqueeze(eta,-1)
        z = torch.bmm(P_mats,xabs_vec - eta_vec)
        z = torch.bmm(torch.transpose(Qr,1,2),z)
        z = torch.linalg.solve(QTPQm,z)
        z = torch.bmm(Qr,z)
        z = z + eta_vec
        z = torch.squeeze(z,-1)
        new_slacks = z[:,-self.n_ineq:]
        new_slacks = 2*torch.relu(slacks) - new_slacks
        return torch.cat((z[:,0:-self.n_ineq],new_slacks),dim=-1)


























class DRSolverQP(nn.Module):
    """
    This solver takes a Quadratic Programming problem in standard form

    min p^T x + x^T Q X
    s.t.
        A x == b
          x >=0


    """
    def __init__(self,f_obj = None,
                 #F_ineq = None,
                 F_eq = None,
                 x_dim = 0,
                 #n_ineq = 0,
                 n_eq = 0,
                 #order = 'first',
                 JF_fixed = False,
                 parm_dim = None,
                 num_steps=3,
                 initial_steps = 1,
                 initial_lb_P = 1e-8,
                 initial_ub_P = 1e8,
                 lb_P = 1.0/2.0,
                 ub_P = 2.0,
                 project_fixedpt=False,
                 slack_mode=False):    # JK: in slack mode, we convert x>=0 to x+s==0 with s>=0
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
        #self.n_ineq = 0
        self.n_eq = n_eq
        self.num_steps = num_steps
        self.initial_steps = initial_steps
        #self.order = order
        self.JF_fixed = JF_fixed
        self.parm_dim = parm_dim
        self.f_obj = f_obj
        #self.F_ineq = F_ineq
        self.F_eq = F_eq

        self.initial_lb_P = initial_lb_P
        self.initial_ub_P = initial_ub_P
        self.lb_P = lb_P
        self.ub_P = ub_P
        self.project_fixedpt = project_fixedpt
        self.slack_mode = slack_mode

        if self.slack_mode:
            # Add in x>=0, and convert it to an equality constraint
            self.n_ineq = self.x_dim
            def F_ineq(x,p):
                x = x[:x_dim]
                return -x
        else:
            # At this point, x>=0 is not part of the model.
            # It manifests later in the projection/ g2 prox which becomes ReLU(x)
            self.n_ineq = 0
            F_ineq = None

        self.F_ineq = F_ineq

        self.n_dim = self.x_dim + self.n_ineq

        if (self.project_fixedpt) & (~self.slack_mode) : input("ERROR: cannot use fixed-point projection without slack mode")
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
        print("self.pid = ")
        input( self.pid    )
        if pid == 0: print( 'ERROR: One of F_eq or F_ineq must be defined')
        if pid == 1:
            def F(xs, parms):
                x = xs[0:self.x_dim]
                return F_eq(xs,parms)
        if pid == 2:
            def F(xs,parms):
                x = xs[0:self.x_dim]
                s = xs[self.x_dim:]
                return self.F_ineq(x,parms) + s
        if pid == 3:
            def Fs_ineq(xs,parms):
                x = xs[0:self.x_dim]
                s = xs[self.x_dim:]
                return torch.cat( (torch.zeros(self.n_eq),  self.F_ineq(x,parms) + s))
            def Fs_eq(xs,parms):
                x = xs[0:self.x_dim]
                return torch.cat( (self.F_eq(x,parms), torch.zeros(self.n_ineq) ))
            def F(xs, parms):
                return Fs_eq(xs,parms) + Fs_ineq(xs,parms)


        self.F = F
        #### Set the Prox of g_1(x,s)
        up_bound  = initial_ub_P#1e8
        low_bound = initial_lb_P#1e-8


        # JK TODO: if lb=ub, use Identity
        self.Pm = mx.ParametricDiagonal(self.n_dim,self.parm_dim,up_bound,low_bound)

        #if self.order == 'first': self.foF = px.FirstOrderObjectiveConstraintComposition(self.f_obj,self.F,self.Pm,JF_fixed = self.JF_fixed,n_dim = self.n_dim, parm_dim = self.parm_dim)
        #if self.order == 'second': self.foF = px.SecondOrderObjectiveConstraintComposition(self.f_obj,self.F,self.Pm,JF_fixed = self.JF_fixed,n_dim = self.n_dim, parm_dim = self.parm_dim)
        self.foF = px.QuadraticObjectiveLinearEqualityComposition(self.f_obj,self.F,self.Pm,n_dim = self.n_dim, parm_dim = self.parm_dim)

        up_bound_Id  = ub_P #2.0
        low_bound_Id = lb_P #1.0/2.0




        self.Pm_2 = mx.ParametricDiagonal(self.n_dim,self.parm_dim,up_bound_Id,low_bound_Id)

        #self.Pm_2 = mx.StaticDiagonal(n_dim = self.n_dim, parm_dim = self.parm_dim)
        #if self.order == 'first':  self.foF_Id =  px.FirstOrderObjectiveConstraintComposition(self.f_obj,self.F,self.Pm_2,JF_fixed = self.JF_fixed,n_dim = self.n_dim, parm_dim = self.parm_dim)
        #if self.order == 'second': self.foF_Id = px.SecondOrderObjectiveConstraintComposition(self.f_obj,self.F,self.Pm_2,JF_fixed = self.JF_fixed,n_dim = self.n_dim, parm_dim = self.parm_dim)
        self.foF_Id = px.EqualityConstrainedQuadratic(self.f_obj,self.F,self.Pm_2,n_dim = self.n_dim, parm_dim = self.parm_dim, slack_mode=slack_mode, n_ineq=self.n_ineq)

        ### Set the Prox of g_2(x,s)

        if self.slack_mode:
            ## define the slack bounds
            upper_bound = 1e3*torch.ones(self.n_dim)
            lower_bound = torch.cat( (-1e3*torch.ones(self.x_dim),torch.zeros(self.n_ineq)))
            def f_upper(parms):
                return upper_bound
            def f_lower(parms):
                return lower_bound
            self.sp = px.BoxConstraint(f_lower,f_upper)
            up_bound_fp = 1e8
            low_bound_fp = 1e-8
            self.fp_Pm = mx.ParametricStateDiagonal(self.n_dim,self.parm_dim,up_bound_fp,low_bound_fp)
        else:
            ## The QP is in standard form, box constraint is x>=0
            def sp(x,parms):
                return torch.relu(x)
            self.sp = sp

    def forward(self,x,parms):
        ## Use zero slack initialization
        if self.n_ineq > 0:
           #add the slack variables
           x = torch.cat((x,torch.zeros((x.shape[0],self.n_ineq))),dim = -1)
        x_k = x
        #Q,R,Md = self.foF.prefactor(x_k,parms)
        for n in range(self.initial_steps):
            y_k = self.sp(x_k,parms)
            z_k = self.foF(2*y_k - x_k,parms)
            #z_k = self.foF.prox(2*y_k - x_k,Q,R,Md,parms)
            x_k_new = x_k + (z_k - y_k)
            x_k = x_k_new

        if self.project_fixedpt:
            x_k = self.FixedPointConditionProjection(x_k[:,:-self.n_ineq],parms)   #JK note this causes crash for n_ineq=0, but project_fixedpt now cannot be used outside slack mode,

        #Q,R,Md = self.foF_Id.prefactor(x_k,parms)                                                      #   which requires n_ineq == x_dim
        for n in range(self.num_steps):
            #start = time.time()
            #print("x_k = ")
            #print( x_k    )

            y_k = self.foF_Id(x_k,parms)

            #print("y_k = ")
            #print( y_k    )

            #print("2*y_k - x_k = ")
            #print( 2*y_k - x_k    )

            z_k = self.sp(2*y_k - x_k,parms)

            #print("z_k = ")
            #print( z_k    )

            #y_k = self.sp(x_k,parms)
            #z_k = self.foF_Id(2*y_k - x_k,parms)

            #z_k = self.foF_Id.prox(2*y_k - x_k,Q,R,Md,parms)
            #y_k = self.foF_Id.prox(x_k,Q,R,Md,parms)
            #z_k = self.sp(2*y_k - x_k,parms)
            x_k_new = x_k + (z_k - y_k)
            x_k = x_k_new
            cnv_gap = z_k - y_k
            #end = time.time()
            #print("Iteration {} takes {}s".format(n,end-start))


        M = x_k_new.shape[1]
        #return x_k_new[:,:-self.n_ineq], cnv_gap   # JK bug note: this indexing causes empty return when n_ineq=0, since -0 = 0
        return x_k_new[:,:M-self.n_ineq], cnv_gap


    def FixedPointConditionProjection(self,x,parms):
        x_init = x
        #add initial slack variables
        xz = torch.cat((x,torch.zeros((x.shape[0],self.n_ineq))),dim = -1)
        xs = torch.vmap(self.F)(xz,parms)
        slacks = -xs[:,-self.n_ineq:]
        x = torch.cat((x,slacks),dim = -1)
        s_plus = torch.cat( (torch.zeros(x.shape[0],self.x_dim),torch.relu(slacks)),dim = -1)
        #if self.order == 'second':
        #    batch_size = x.shape[0]
        #    Hx = self.foF.H_vec(x,parms)
        #    Id_batch = torch.tile(torch.unsqueeze(torch.eye(self.n_dim),dim=0),(batch_size,1,1))
        #    Md = (self.foF.gamma/2)*Id_batch + Hx
        #    grads = torch.linalg.solve(Md,self.foF.f_grad(x,parms))
        #if self.order == 'first':
        #    grads = (self.foF.gamma/2)*self.foF.f_grad(x,parms)

        batch_size = x.shape[0]
        Hx = self.foF.H_vec(x,parms)
        Id_batch = torch.tile(torch.unsqueeze(torch.eye(self.n_dim),dim=0),(batch_size,1,1))
        Md = (self.foF.gamma/2)*Id_batch + Hx
        grads = torch.linalg.solve(Md,self.foF.f_grad(x,parms))
        eta = s_plus + grads
        JFx = self.foF.JF(x,parms)
        ### Take a QR decomposition of the Jacobian
        with torch.no_grad():
            Q, R  = torch.linalg.qr(torch.transpose(JFx,1,2),mode = 'complete')
            null_dim = Q.shape[-1] - R.shape[-1]
            R = R[:,:-null_dim,:]
            Qr = Q[:,:,:-null_dim]
            Qn = Q[:,:,-null_dim:]
        P_mats = vmap(self.fp_Pm)(x,parms)
        QTPQm = torch.bmm(torch.transpose(Qr,1,2),torch.bmm(P_mats,Qr))
        # Compute the oblique projection
        xabs = torch.cat( (x_init,torch.abs(slacks)),dim = -1)
        xabs_vec = torch.unsqueeze(xabs,-1)
        eta_vec = torch.unsqueeze(eta,-1)
        z = torch.bmm(P_mats,xabs_vec - eta_vec)
        z = torch.bmm(torch.transpose(Qr,1,2),z)
        z = torch.linalg.solve(QTPQm,z)
        z = torch.bmm(Qr,z)
        z = z + eta_vec
        z = torch.squeeze(z,-1)
        new_slacks = z[:,-self.n_ineq:]
        new_slacks = 2*torch.relu(slacks) - new_slacks
        return torch.cat((z[:,0:-self.n_ineq],new_slacks),dim=-1)







"""
# JK practice converting to basic QP case
# no extra slack variables
class DRSolverQP(nn.Module):

    #Implementation of Douglas Rachford (DR) Iterations for corrections of solution estimates for problems of the form
    #min f(x)
    #subject to:
    #F_ineq(x) <= 0
    #F_eq(x)= 0
    #
    #The problem is reformulated as
    #
    #min f(x)
    #subject to:
    #F(x,s) = 0
    #s>=0
    #
    #for slack variables s, and F(x,s) defined as
    #
    #F(x,s) = [ F_eq(x) ; F_ineq(x) + s ]
    #
    #DR is an operator splitting approach, here applied to the splitting
    #
    #min g_1(x,s) + g_2(x,s)
    #
    #with
    # g_1(x,s) = f(x) + i_{ (x,s) : F(x,s) = 0}
    # g_2(x) = i_{ s : s>=0 }

    #where i_{S} is the indicator function on set S.


    def __init__(self,f_obj = None,
                 F_ineq = None,
                 F_eq = None,
                 x_dim = 0,
                 n_ineq = 0,
                 n_eq = 0,
                 order = 'first',
                 JF_fixed = False,
                 parm_dim = None,
                 num_steps=3,
                 initial_steps = 1,
                 initial_lb_P = 1e-8,
                 initial_ub_P = 1e8,
                 lb_P = 1.0/2.0,
                 ub_P = 2.0,
                 project_fixedpt=True):

        #:param f_obj: functorch compatible function) a parameterized function f with input of the form (x,parms) where parms is a tensor that matches the batch of x, and has last dim parm features.
        #          f is defined unbatched, the method will call vmap, to "raise" f to batch dim
        #          gives the objective to be optimized
        #:param F_ineq:(functorch compatible function) a parameterized function F with input of the form (x, parms) where parms is a tensor that matches the batch of x, and has last dim parm features.
        #          F is defined unbatched, the method will call vmap, to "raise" f to batch dim
        #          gives the inequality constraints to satisfy, F_ineq(x) <= 0
        #:param F_eq:(functorch compatible function) a parameterized function F with input of the form (x, parms) where parms is a tensor that matches the batch of x, and has last dim parm features.
        #          F is defined unbatched, the method will call vmap, to "raise" f to batch dim
        #          gives the equality constraints to satisfy, F_eq(x) = 0
        #:param x_dim: (int) dimension of the primal variables
        #:param n_ineq: (int) number of inequality constraints
        #:param n_eq: (int) number of equality constraints
        #:param order: (str) one of {'first','second'} the order of the approximation used for f_obj
        #:param JF_fixed: (Bool) Indicates if the Jacobian of F should be computed at each iteration. Default is False, if True Jacobian of F will be precomputed at x=0, parms = 0
        #:param parm_dim: (int) the dimension of parms for precomputing Jacobian
        #:param num_steps: (int) number of iteration steps for the Douglas Rachford method using Identity metric
        #:param initial_steps: (int) number of learned metric DR steps to take, default is one.

        super().__init__()
        self.x_dim = x_dim
        self.n_ineq = n_ineq
        self.n_eq = n_eq
        self.num_steps = num_steps
        self.initial_steps = initial_steps
        self.order = order
        self.JF_fixed = JF_fixed
        self.parm_dim = parm_dim
        self.n_dim = self.x_dim #+ self.n_ineq
        self.f_obj = f_obj
        self.F_ineq = F_ineq
        self.F_eq = F_eq

        self.initial_lb_P = initial_lb_P
        self.initial_ub_P = initial_ub_P
        self.lb_P = lb_P
        self.ub_P = ub_P
        self.project_fixedpt = project_fixedpt

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
        if pid != 1: print('ERROR: Only F_eq must be defined')
        if pid == 1:
            def F(xs, parms):
                return F_eq(xs,parms)

        self.F = F
        #### Set the Prox of g_1(x,s)
        up_bound  = initial_ub_P#1e8
        low_bound = initial_lb_P#1e-8

        self.Pm = mx.ParametricDiagonal(self.n_dim,self.parm_dim,up_bound,low_bound)

        #if self.order == 'first': self.foF = px.FirstOrderObjectiveConstraintComposition(self.f_obj,self.F,self.Pm,JF_fixed = self.JF_fixed,n_dim = self.n_dim, parm_dim = self.parm_dim)
        self.foF = px.SecondOrderObjectiveConstraintComposition(self.f_obj,self.F,self.Pm,JF_fixed = self.JF_fixed,n_dim = self.n_dim, parm_dim = self.parm_dim)
        up_bound_Id  = ub_P #2.0
        low_bound_Id = lb_P #1.0/2.0

        self.Pm_2 = mx.ParametricDiagonal(self.n_dim,self.parm_dim,up_bound_Id,low_bound_Id)

        #if self.order == 'first':  self.foF_Id =  px.FirstOrderObjectiveConstraintComposition(self.f_obj,self.F,self.Pm_2,JF_fixed = self.JF_fixed,n_dim = self.n_dim, parm_dim = self.parm_dim)
        self.foF_Id = px.SecondOrderObjectiveConstraintComposition(self.f_obj,self.F,self.Pm_2,JF_fixed = self.JF_fixed,n_dim = self.n_dim, parm_dim = self.parm_dim)
        ### Set the Prox of g_2(x,s)
        ## define the slack bounds
        #upper_bound = 1e3*torch.ones(self.n_dim)
        #lower_bound = torch.cat( (-1e3*torch.ones(self.x_dim),torch.zeros(self.n_ineq)))
        #def f_upper(parms):
        #    return upper_bound
        #def f_lower(parms):
        #    return lower_bound
        #self.sp = px.BoxConstraint(f_lower,f_upper)
        up_bound_fp = 1e8
        low_bound_fp = 1e-8
        self.fp_Pm = mx.ParametricStateDiagonal(self.n_dim,self.parm_dim,up_bound_fp,low_bound_fp)

    def forward(self,x,parms):
        ## Use zero slack initialization
        #if self.n_ineq > 0:
           #add the slack variables
           #x = torch.cat((x,torch.zeros((x.shape[0],self.n_ineq))),dim = -1)
        x_k = x
        for n in range(self.initial_steps):
            #y_k = self.sp(x_k,parms)
            y_k = torch.relu(x_k)
            z_k = self.foF(2*y_k - x_k,parms)
            x_k_new = x_k + (z_k - y_k)
            x_k = x_k_new

        # No fixed point projection

        for n in range(self.num_steps):
            y_k = torch.relu(x_k)
            z_k = self.foF_Id(2*y_k - x_k,parms)
            x_k_new = x_k + (z_k - y_k)
            x_k = x_k_new
            cnv_gap = z_k - y_k

        print("x_k_new[:,:-self.n_ineq]")
        print( x_k_new[:,:-self.n_ineq] )

        return x_k_new[:,:-self.n_ineq], cnv_gap

"""
