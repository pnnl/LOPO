
import torch
import torch.nn as nn

import Prox as px

from importlib import reload
reload(px)


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
    def __init__(self,f_obj = None, F_ineq = None, F_eq = None, x_dim = 0, n_ineq = 0, n_eq = 0,order = 'first',JF_fixed = False, parm_dim = None,num_steps=1):
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
        :param num_steps: (int) number of iteration steps for the Douglas Rachford method
        """
        super().__init__()
        self.f_obj = f_obj
        self.F_ineq = F_ineq
        self.F_eq = F_eq
        self.x_dim = x_dim
        self.n_ineq = n_ineq
        self.n_eq = n_eq
        self.num_steps = num_steps
        self.order = order
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
        self.JF_fixed = JF_fixed
        self.n_dim = self.x_dim + self.n_eq + self.n_ineq
        self.parm_dim = parm_dim
        if self.order == 'first': self.foF = px.FirstOrderObjectiveConstraintComposition(self.f_obj,self.F,JF_fixed = self.JF_fixed,n_dim = self.n_dim, parm_dim = self.parm_dim)
        if self.order == 'second': self.foF = px.SecondOrderObjectiveConstraintComposition(self.f_obj,self.F,JF_fixed = self.JF_fixed,n_dim = self.n_dim, parm_dim = self.parm_dim)
        ### Set the Prox of g_2(x,s)
        ## define the slack bounds
        upper_bound = 1e2*torch.ones(self.n_dim)
        lower_bound = torch.cat( (-1e2*torch.ones(self.x_dim + self.n_eq ),torch.zeros(self.n_ineq)))
        def f_upper(parms):
            return upper_bound
        def f_lower(parms):
            return lower_bound
        self.sp = px.BoxConstraint(f_lower,f_upper)
    def forward(self,x,parms):
        x = self.SlackHotStart(x,parms)
        x_k = x  
        for n in range(self.num_steps):
            y_k = self.sp(x_k,parms)
            z_k = self.foF(2*y_k - x_k,parms)
            x_k_new = x_k + (z_k - y_k)
            x_k = x_k_new
        return x_k_new[:,:-self.n_ineq]
    def SlackHotStart(self,x,parms):
        x_init = x
        #add initial slack variables
        xz = torch.cat((x,torch.zeros((x.shape[0],self.n_ineq))),dim = -1)
        xs = torch.vmap(self.F)(xz,parms)
        slacks = -xs[:,-self.n_ineq:]
        x = torch.cat((x,slacks),dim = -1)
        s_plus = torch.cat( (torch.zeros(x.shape[0],self.x_dim),torch.relu(slacks)),dim = -1)
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
        xs_plus = torch.cat( ( 1e3*torch.ones(x_init.shape),torch.relu(slacks)),dim = -1)
        P_diags = torch.abs(xs_plus) + 1e-3
        P_mats = torch.diag_embed(P_diags)
        QTPQ = torch.bmm(torch.transpose(Qr,1,2),torch.bmm(P_mats,Qr))
        # Compute the oblique projection
        xabs = torch.cat( (x_init,torch.abs(slacks)),dim = -1)
        xabs_vec = torch.unsqueeze(xabs,-1)
        eta_vec = torch.unsqueeze(eta,-1)
        z = torch.bmm(P_mats,xabs_vec - eta_vec)
        z = torch.bmm(torch.transpose(Qr,1,2),z)
        z = torch.linalg.solve(QTPQ,z)
        z = torch.bmm(Qr,z)
        z = z + eta_vec
        z = torch.squeeze(z,-1)
        new_slacks = z[:,-self.n_ineq:]
        new_slacks = 2*torch.relu(slacks) - new_slacks
        return torch.cat((z[:,0:-self.n_ineq],new_slacks),dim=-1)
    

