
import torch
import torch.nn as nn
from torch.func import grad
from torch.func import vmap
from torch.func import jacrev
from torch.func import hessian

class ParametericDRSolver(nn.Module):
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

    This Routine uses a three stage approach:
        1. Initial DR iteration
        2. Fixed Point Projection
        3. Final DR iteration
    A different metric is learned for each stage. The learned metric is a neural network mapping from proglem parameters to a positive definite matrix.
        1. A Metric is learned and applied for the prox of g_1 in DR iterations to improve initial convergence
        2. A metric is learned and applied for a projection onto the DR fixed point conditions, for primal and slack estimation.
        3. A metric is learned and applied for the prox of g_1 in DR iterations to improve convergence
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
                 lb_P = 1.0/5.0,
                 ub_P = 5.0,
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
        '''Setup Stage 1, initial DR iterations'''
        #Initialize metric for prox of g_1(x,s)
        P_up_bound  = initial_ub_P
        P_low_bound = initial_lb_P
        self.Pm = ParametricDiagonal(self.n_dim,self.parm_dim,P_up_bound,P_low_bound)
        self.foF = ObjectiveConstraintComposition(self.f,self.F,self.Pm,order = self.order,JF_fixed = self.JF_fixed,n_dim = self.n_dim, parm_dim = self.parm_dim)
        '''Setup Stage 2, fixed point projection'''
        P_up_bound  = initial_ub_P
        P_low_bound = initial_lb_P
        self.fp_Pm = ParametricStateDiagonal(self.n_dim,self.parm_dim,P_up_bound,P_low_bound)
        '''Setup Stage 3, final DR iterations'''
        P_up_bound_final  = ub_P 
        P_low_bound_final = lb_P 
        self.Pm_final = ParametricDiagonal(self.n_dim,self.parm_dim,P_up_bound_final,P_low_bound_final)
        self.foF_final =  ObjectiveConstraintComposition(self.f,self.F,self.Pm_final,order = self.order,JF_fixed = self.JF_fixed,n_dim = self.n_dim, parm_dim = self.parm_dim)
        ''' Define Slacks, and bounds for DR iterations'''
        ## define the slack bounds
        upper_bound = 1e3*torch.ones(self.n_dim)
        lower_bound = torch.cat( (-1e3*torch.ones(self.x_dim),torch.zeros(self.n_ineq)))
        def f_upper(parms):
            return upper_bound
        def f_lower(parms):
            return lower_bound
        self.sp = BoxConstraint(f_lower,f_upper)
    def forward(self,x,parms):
        #add the slack variables
        if self.n_ineq > 0:
           x = torch.cat((x,torch.zeros((x.shape[0],self.n_ineq))),dim = -1)
        x_k = x
        '''Stage 1, initial DR iterations'''
        for n in range(self.initial_steps):
            y_k = self.sp(x_k,parms)
            z_k = self.foF(2*y_k - x_k,parms)
            x_k_new = x_k + (z_k - y_k)
            x_k = x_k_new
        '''Stage 2, fixed point projection'''
        x_k = self.FixedPointConditionProjection(x_k[:,:-self.n_ineq],parms)
        '''Stage 3, final DR iterations'''
        for n in range(self.num_steps):
            y_k = self.sp(x_k,parms)
            z_k = self.foF_final(2*y_k - x_k,parms)
            x_k_new = x_k + (z_k - y_k)
            x_k = x_k_new
            cnv_gap = z_k - y_k
        return x_k_new[:,:-self.n_ineq], cnv_gap
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

'''
PROX DEFINITIONS
'''
class ObjectiveConstraintComposition(torch.nn.Module):
    '''
    Computes an approximation of the prox operator of the sum
    f(x) + i_{ x : F(x) = 0 }
    where F: R^{n}->R^{m}, and m <= n , and is assumed to be differentiable everywhere
          i_{} is the indicator function
          f: R^{n}-> R is scalar valued and assumed to be differentiable everywhere
    For a given x the prox operator is computed for one of the approximations,
    order = 'first':
    f(x) + grad_f(x)^T(y - x) + i_{ y: F(x) + J_F(x)*(y - x) = 0 }
    order = 'second':
    f(x) + grad_f(x)^T(y - x) + (y-x)^T H_f(x) (y-x) + i_{ y: F(x) + J_F(x)*(y - x) = 0 }
    with grad_f(x) the gradient of f at x, H_f(x) the hessian of f at x, and  J_F(x) the Jacobian of F at x
    '''
    def __init__(self,f,F,metric,order = 'first',JF_fixed = False,n_dim = None, parm_dim = None):
        '''
        :param f: (functorch compatible function) a parameterized function f with input of the form (x,parms) where parms is a tensor that matches the batch of x, and has last dim parm features.
                  f is defined unbatched, the method will call vmap, to "raise" f to batch dim
        :param F:(functorch compatible function) a parameterized function F with input of the form (x, parms) where parms is a tensor that matches the batch of x, and has last dim parm features.
                  F is defined unbatched, the method will call vmap, to "raise" f to batch dim
        :param metric: (function) a parameterized function with input of the form (x, parms) where parms is a tensor that matches the batch of x, and has last dim parm features. Returns a positive definie
                  matrix of dimension dim(x) X dim(x). Can be defined unbatched, method will call vmap to 'raise' to batch dim.
        :param order: (str) one of {'first','second'} the order of the approximation used for f
        :param JF_fixed: (Bool) Indicates if the Jacobian of F should be computed at each iteration. Default is False, if True Jacobian of F will be precomputed at x=0, parms = 0
        :param n_dim: (int) the dimension of x for precomputing Jacobian
        :param parm_dim: (int) the dimension of parms for precomputing Jacobian
        '''
        super().__init__()
        self.f = vmap(f)
        self.f_grad = vmap(grad(f,argnums = 0))
        self.H_vec = vmap(hessian(f,argnums = 0))
        self.F = vmap(F)
        self.JF = vmap(jacrev(F,argnums = 0))
        self.gamma = 2.0
        self.metric = metric
        self.JF_fixed = JF_fixed
        self.order = order
        if self.JF_fixed == True:
            jx = torch.zeros((1,n_dim),dtype = torch.float32)
            jp = torch.zeros((1,parm_dim),dtype = torch.float32)
            JFx = self.JF(jx,jp)
            Qc, Rc  = torch.linalg.qr(torch.transpose(JFx,1,2),mode = 'complete')
            null_dim = Qc.shape[-1] - Rc.shape[-1]
            Rr = Rc[:,:-null_dim,:]
            Qr = Qc[:,:,:-null_dim]
            Qn = Qc[:,:,-null_dim:]
            self.JFx = JFx
            self.Qr = Qr
            self.Qn = Qn
            self.Rr = Rr
    def forward(self, x, parms):
        if self.JF_fixed == False:
            with torch.no_grad():
                JFx = self.JF(x,parms)
                ### Take a QR decomposition of the Jacobian
                Qc, Rc  = torch.linalg.qr(torch.transpose(JFx,1,2),mode = 'complete')
                null_dim = Qc.shape[-1] - Rc.shape[-1]
                Rr = Rc[:,:-null_dim,:]
                Qr = Qc[:,:,:-null_dim]
                Qn = Qc[:,:,-null_dim:]
        if self.JF_fixed == True:
            batch_size = x.shape[0]
            JFx = torch.tile(self.JFx,(batch_size,1,1))
            Qr = torch.tile(self.Qr,(batch_size,1,1))
            Qn = torch.tile(self.Qn,(batch_size,1,1))
            Rr = torch.tile(self.Rr,(batch_size,1,1))
        Fx = self.F(x,parms)
        # Compute the RHS
        Fx_mat = torch.unsqueeze(Fx,dim = -1)
        x_mat = torch.unsqueeze(x,dim = -1)
        b = -Fx_mat + torch.bmm(JFx,x_mat)
        #### Find a solution to RHS
        Rr_T = torch.transpose(Rr,1,2)
        zeta_r = torch.linalg.solve_triangular(Rr_T,b,upper=False,left = True)
        zeta = torch.bmm(Qr,zeta_r)
        # Compute the gradient of f
        fg = torch.unsqueeze(self.f_grad(x,parms),dim = -1)
        if self.order =='first': 
            ### Compute the gradient step and Projection Operation
            Pm = vmap(self.metric)(x,parms)
            QtPQ = torch.bmm(torch.bmm(torch.transpose(Qn,1,2),Pm),Qn)
            zq = torch.bmm(Pm,x_mat - zeta) - (self.gamma/2)*fg
            zq = torch.bmm(torch.transpose(Qn,1,2),zq)
            zq = torch.linalg.solve(QtPQ,zq)
            zq = torch.bmm(Qn,zq)
            x_new =  zq + zeta
        if self.order =='second':
            # Compute the Hessian of f
            Hf = self.H_vec(x,parms)
            ### Compute the gradient step and Projection Operation
            Pm = vmap(self.metric)(x,parms)
            Md =  (self.gamma/2)*Pm + Hf
            QTMdQ = torch.bmm(torch.bmm(torch.transpose(Qn,1,2),Md),Qn)
            zq = torch.bmm(Md,x_mat - zeta) - (self.gamma/2)*fg
            zq = torch.bmm(torch.transpose(Qn,1,2),zq)
            zq = torch.linalg.solve(QTMdQ,zq)
            zq = torch.bmm(Qn,zq)
            x_new =  zq + zeta
        return torch.squeeze(x_new,dim = -1)
    
class BoxConstraint(torch.nn.Module):
    '''
    Computes the projection onto the box constraints l_b <= x <= u_b for constants l_b and u_b.
    '''
    def __init__(self,f_lower_bound,f_upper_bound):
        '''
        :param f_lower_bound: (function) has form f(parms), returns tensor of lower bound constraints, is defined unbatched, will be 'raised' with vmap.
        :param f_upper_bound: (function) has form f(parms), returns tensor of upper bound constraints, is defined unbatched, will be 'raised' with vmap.
        '''
        super().__init__()
        self.lower_bound_func = vmap(f_lower_bound)
        self.upper_bound_func = vmap(f_upper_bound)
    def forward(self, x, parms):
        l_b = self.lower_bound_func(parms)
        u_b = self.upper_bound_func(parms)
        return l_b + torch.relu( u_b - l_b - torch.relu(u_b - x ))

'''
METRIC DEFINITIONS
'''
class ParametricDiagonal(torch.nn.Module):
    '''
    A neural network mapping from problem parameters to a positive definite diagonal matrix
    '''
    def __init__(self,n_dim,parm_dim,upper_bound,lower_bound):
        '''
        :param n_dim: dimenison of output matrix
        :param parm_dim: dimension of parameter inputs
        :param upper_bound: upper bound on diagonal entries of matrix output
        :param lower_bound: lower bound on diagonal entries of matrix output
        '''
        super().__init__()
        self.n_dim = n_dim
        self.P_diag_upper_bound = upper_bound
        self.P_diag_lower_bound = lower_bound
        self.parm_dim = parm_dim
        self.hidden_dim = 10*self.n_dim
        self.DiagMap = nn.Sequential(
          nn.Linear(self.parm_dim,self.hidden_dim),
          nn.ReLU(),
          nn.Linear(self.hidden_dim,self.hidden_dim),
          nn.ReLU(),
          nn.Linear(self.hidden_dim,self.n_dim),
        )
    def forward(self,x,parms):
        Pd = self.DiagMap(parms)
        P_diag = self.P_diag_lower_bound + torch.sigmoid(Pd)*(self.P_diag_upper_bound - self.P_diag_lower_bound)
        Pm = torch.diag(P_diag)
        return Pm

class ParametricStateDiagonal(torch.nn.Module):
    '''
    A neural network mapping from primal estimates and problem parameters to a positive definite diagonal matrix
    '''
    def __init__(self,n_dim,parm_dim,upper_bound,lower_bound):
        '''
        :param n_dim: dimenison of output matrix
        :param parm_dim: dimension of parameter inputs
        :param upper_bound: upper bound on diagonal entries of matrix output
        :param lower_bound: lower bound on diagonal entries of matrix output
        '''
        super().__init__()
        self.n_dim = n_dim
        self.P_diag_upper_bound = upper_bound
        self.P_diag_lower_bound = lower_bound
        self.input_dim = parm_dim + n_dim
        self.hidden_dim = 10*self.n_dim
        self.DiagMap = nn.Sequential(
          nn.Linear(self.input_dim,self.hidden_dim),
          nn.ReLU(),
          nn.Linear(self.hidden_dim,self.hidden_dim),
          nn.ReLU(),
          nn.Linear(self.hidden_dim,self.n_dim),
        )
    def forward(self,x,parms):
        xsp = torch.cat((x,parms),dim = -1)
        Pd = self.DiagMap(xsp)
        P_diag = self.P_diag_lower_bound + torch.sigmoid(Pd)*(self.P_diag_upper_bound - self.P_diag_lower_bound)
        Pm = torch.diag(P_diag)
        return Pm









