
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
    Implementation of Douglas Rachford (DR) Iterations for corrections of solution estimates for problems of the form
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
            def F(xs, parms):
                return F_eq(xs,parms)
        if pid == 2:
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
        self.F = F
        #### Set the Prox of g_1(x,s)
        up_bound  = initial_ub_P#1e8
        low_bound = initial_lb_P#1e-8


        # JK TODO: if lb=ub, use Identity
        self.Pm = mx.ParametricDiagonal(self.n_dim,self.parm_dim,up_bound,low_bound)

        if self.order == 'first': self.foF = px.FirstOrderObjectiveConstraintComposition(self.f_obj,self.F,self.Pm,JF_fixed = self.JF_fixed,n_dim = self.n_dim, parm_dim = self.parm_dim)
        if self.order == 'second': self.foF = px.SecondOrderObjectiveConstraintComposition(self.f_obj,self.F,self.Pm,JF_fixed = self.JF_fixed,n_dim = self.n_dim, parm_dim = self.parm_dim)
        up_bound_Id  = ub_P #2.0
        low_bound_Id = lb_P #1.0/2.0




        self.Pm_2 = mx.ParametricDiagonal(self.n_dim,self.parm_dim,up_bound_Id,low_bound_Id)

        #self.Pm_2 = mx.StaticDiagonal(n_dim = self.n_dim, parm_dim = self.parm_dim)
        if self.order == 'first':  self.foF_Id =  px.FirstOrderObjectiveConstraintComposition(self.f_obj,self.F,self.Pm_2,JF_fixed = self.JF_fixed,n_dim = self.n_dim, parm_dim = self.parm_dim)
        if self.order == 'second': self.foF_Id = px.SecondOrderObjectiveConstraintComposition(self.f_obj,self.F,self.Pm_2,JF_fixed = self.JF_fixed,n_dim = self.n_dim, parm_dim = self.parm_dim)
        ### Set the Prox of g_2(x,s)
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

    def forward(self,x,parms):
        ## Use zero slack initialization
        if self.n_ineq > 0:
           #add the slack variables
           x = torch.cat((x,torch.zeros((x.shape[0],self.n_ineq))),dim = -1)
        x_k = x
        for n in range(self.initial_steps):
            y_k = self.sp(x_k,parms)
            z_k = self.foF(2*y_k - x_k,parms)
            x_k_new = x_k + (z_k - y_k)
            x_k = x_k_new

        if self.project_fixedpt:
            x_k = self.FixedPointConditionProjection(x_k[:,:-self.n_ineq],parms)

        for n in range(self.num_steps):
            y_k = self.sp(x_k,parms)
            z_k = self.foF_Id(2*y_k - x_k,parms)
            x_k_new = x_k + (z_k - y_k)
            x_k = x_k_new
            cnv_gap = z_k - y_k
        return x_k_new[:,:-self.n_ineq], cnv_gap   # JK bug note: this indexing causes return


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
                 final_steps = 0,
                 gamma = 2.0,
                 initial_lb_P = 1e-8,
                 initial_ub_P = 1e8,
                 lb_P = 1.0/2.0,
                 ub_P = 2.0,
                 project_fixedpt=False,
                 slack_mode=False,        # JK: in slack mode, we convert x>=0 to x+s==0 with s>=0
                 precision=torch.float64,
                 fixed_metric = None):    # Makes it so the steps taken during the main iterations of num_steps (n_opt_steps) follow the metric given

        super().__init__()
        self.x_dim = x_dim
        #self.n_ineq = 0
        self.n_eq = n_eq
        self.num_steps = num_steps
        self.initial_steps = initial_steps
        self.final_steps = final_steps
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
        self.precision = precision

        self.gamma = gamma

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

        if (self.project_fixedpt) & (~self.slack_mode) : print("ERROR: cannot use fixed-point projection without slack mode"); quit()
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
        self.Pm_1 = mx.ParametricDiagonal(self.n_dim,self.parm_dim,up_bound,low_bound)

        #if self.order == 'first': self.foF = px.FirstOrderObjectiveConstraintComposition(self.f_obj,self.F,self.Pm,JF_fixed = self.JF_fixed,n_dim = self.n_dim, parm_dim = self.parm_dim)
        #if self.order == 'second': self.foF = px.SecondOrderObjectiveConstraintComposition(self.f_obj,self.F,self.Pm,JF_fixed = self.JF_fixed,n_dim = self.n_dim, parm_dim = self.parm_dim)
        self.foF_1 = px.ModularQuadraticObjectiveLinearEqualityComposition(self.f_obj,self.F,self.Pm_1,JF_fixed=self.JF_fixed,n_dim = self.n_dim, parm_dim = self.parm_dim, precision = self.precision, gamma=self.gamma)

        up_bound_Id  = ub_P #2.0
        low_bound_Id = lb_P #1.0/2.0

        if fixed_metric == None:
            self.Pm_2  = mx.ParametricDiagonal(self.n_dim,self.parm_dim,up_bound_Id,low_bound_Id)
        else:
            self.Pm_2 =  mx.Identity(self.n_dim,self.parm_dim,P_d=fixed_metric)

        self.Pm_Id = mx.Identity(self.n_dim,self.parm_dim)

        #self.Pm_2 = mx.StaticDiagonal(n_dim = self.n_dim, parm_dim = self.parm_dim)
        #if self.order == 'first':  self.foF_Id =  px.FirstOrderObjectiveConstraintComposition(self.f_obj,self.F,self.Pm_2,JF_fixed = self.JF_fixed,n_dim = self.n_dim, parm_dim = self.parm_dim)
        #if self.order == 'second': self.foF_Id = px.SecondOrderObjectiveConstraintComposition(self.f_obj,self.F,self.Pm_2,JF_fixed = self.JF_fixed,n_dim = self.n_dim, parm_dim = self.parm_dim)

        self.foF_2a = px.ModularQuadraticObjectiveLinearEqualityComposition(self.f_obj,self.F,self.Pm_2,JF_fixed=self.JF_fixed,n_dim = self.n_dim, parm_dim = self.parm_dim, precision = self.precision,gamma=self.gamma)
        self.foF_2b = px.EqualityConstrainedQuadratic(self.f_obj,self.F,self.Pm_2,n_dim = self.n_dim, parm_dim = self.parm_dim, slack_mode=slack_mode, n_ineq=self.n_ineq,gamma=self.gamma)


        self.foF_Id = px.ModularQuadraticObjectiveLinearEqualityComposition(self.f_obj,self.F,self.Pm_Id,JF_fixed=self.JF_fixed,n_dim = self.n_dim, parm_dim = self.parm_dim, precision = self.precision,gamma=self.gamma)


        def cvx_qp(N,Q):
            x = cp.Variable(N)
            p = cp.Parameter(N)
            z = cp.Parameter(N)
            constraints = [  cp.sum(x) == 1  ]
            problem  = cp.Problem(cp.Minimize(  p @ x  + (1.0/2.0)*cp.quad_form(x,Q) + (1.0/2.0*gamma)*cp.quad_form(x-z,I)   ),  constraints)  #
            qp_cvxlayer = CvxpyLayer(problem, parameters=[p,z], variables=[x])
            qp_cvxlayer_post = lambda p,z: qp_cvxlayer(p,z)[0]
            return qp_cvxlayer_post
        self.foF_cvx =


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
            ## The QP is in standard form Ax = b w/ x>=0, box constraint is x>=0
            def sp(x,parms):
                return torch.relu(x)
            self.sp = sp

    def forward(self,x,parms):

        gap_resi_list = []

        ## Use zero slack initialization
        if self.n_ineq > 0:
           #add the slack variables
           s = x # since x-s = 0, s>=0
           x = torch.cat((x,s),dim = -1)
           #x = torch.cat((x,torch.zeros((x.shape[0],self.n_ineq))),dim = -1)
        x_k = x
        J_factors, H_factors = self.foF_1.prefactor(x_k,parms)
        for n in range(self.initial_steps):
            y_k = self.sp(x_k,parms)
            #z_k = self.foF(2*y_k - x_k,parms)
            z_k = self.foF_1(2*y_k - x_k,parms,J_factors,H_factors)
            x_k_new = x_k + (z_k - y_k)
            x_k = x_k_new

            with torch.no_grad():
                gap_resi_list.append( torch.norm(z_k - y_k,p=2,dim=1) )

        if self.project_fixedpt:
            x_k = self.FixedPointConditionProjection(x_k[:,:-self.n_ineq],parms)   #JK this crashes for n_ineq=0, but now project_fixedpt cannot be used outside slack mode,

        J_factors, H_factors = self.foF_2a.prefactor(x_k,parms)                                                      #   which requires n_ineq == x_dim.
        for n in range(self.num_steps):
            y_k = self.sp(x_k,parms)
            z_kb = self.foF_2b(2*y_k - x_k,parms)
            z_k = self.foF_2a(2*y_k - x_k,parms,J_factors,H_factors)

            print("z_k = ")
            print( z_k    )
            print("z_kb = ")
            print( z_kb    )
            print("z_k.sum(1) = ")
            print( z_k.sum(1)    )
            print("z_kb.sum(1) = ")
            print( z_kb.sum(1)    )
            input("waiting")

            x_k_new = x_k + (z_k - y_k)
            x_k = x_k_new
            with torch.no_grad():
                gap_resi_list.append( torch.norm(z_k - y_k,p=2,dim=1) )

        J_factors, H_factors = self.foF_Id.prefactor(x_k,parms)
        for n in range(self.final_steps):
            y_k = self.sp(x_k,parms)
            #z_k = self.foF_Id(2*y_k - x_k,parms)
            z_k = self.foF_Id(2*y_k - x_k,parms,J_factors,H_factors)

            #y_k = self.foF_Id.prox(x_k,Q,R,Md,parms)
            #z_k = self.sp(2*y_k - x_k,parms)
            x_k_new = x_k + (z_k - y_k)
            x_k = x_k_new

            with torch.no_grad():
                gap_resi_list.append( torch.norm(z_k - y_k,p=2,dim=1) )


        cnv_gap = z_k - y_k
        x_k_new = self.sp(x_k_new,parms)

        cnv_gap_residuals = torch.stack( gap_resi_list ).T # each row is the cnv gap across iterations, for one sample

        M = x_k_new.shape[1]
        #return x_k_new[:,:-self.n_ineq], cnv_gap   # JK bug note: this indexing causes empty return when n_ineq=0, since -0 = 0
        return x_k_new[:,:M-self.n_ineq], cnv_gap, cnv_gap_residuals


    def FixedPointConditionProjection(self,x,parms):

        x = x.to(self.precision)
        parms = parms.to(self.precision)


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
        Hx = self.foF_1.H_vec(x,parms)
        Id_batch = torch.tile(torch.unsqueeze(torch.eye(self.n_dim),dim=0),(batch_size,1,1))
        Md = (self.foF_1.gamma/2)*Id_batch + Hx
        grads = torch.linalg.solve(Md,self.foF_1.f_grad(x,parms))
        eta = s_plus + grads
        JFx = self.foF_1.JF(x,parms)
        ### Take a QR decomposition of the Jacobian
        with torch.no_grad():
            Q, R  = torch.linalg.qr(torch.transpose(JFx,1,2),mode = 'complete')
            null_dim = Q.shape[-1] - R.shape[-1]
            R = R[:,:-null_dim,:]
            Qr = Q[:,:,:-null_dim]
            Qn = Q[:,:,-null_dim:]
        P_mats = vmap(self.fp_Pm)(x.float(),parms.float()).to(self.precision)
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
