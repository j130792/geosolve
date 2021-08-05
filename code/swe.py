#Global imports
from firedrake import *
import numpy as np
import matplotlib.pylab as plt
import scipy as sp

#local
import refd2 as refd

class problem(object):
    def __init__(self,N,M,degree,T):
        self.mlength = 40
        self.degree = degree
        self.dim = 2
        
        self.c = 1
        self.f = 0.1
        
        self.N = N
        self.M = M
        self.T = T
        self.dt = float(T)/N
        self.mesh = PeriodicSquareMesh(self.M,self.M,self.mlength)

    def function_space(self,mesh):
        RT = FunctionSpace(mesh,'RT',self.degree)
        DG = FunctionSpace(mesh,'DG',self.degree-1)
        return MixedFunctionSpace((RT,DG))

    def exact(self,x,y,t):
        """
        An "exact" initial condition for the linear KdV equation
        """
        u = as_vector((0,0))
        rho = 10 * exp(- ((x-20)**2 + (y-20)**2)/20**2)
        dict = {'u': u,
                'rho': rho}
        return rho, dict
        

def linforms(N=100,M=5,degree=1,T=10,zinit=None):
    #set up problem class
    prob = problem(N=N,M=M,degree=degree,T=T)
    #Set up finite element stuff
    mesh = prob.mesh
    Z = prob.function_space(mesh)
    
    #set up DG stuff
    n = FacetNormal(mesh)
    
    #Set up initial conditions
    z0 = Function(Z)
    u0, rho0 = z0.split()

    t = 0.
    x = SpatialCoordinate(Z.mesh())
    
    if zinit==None:
        rho0.assign(project(prob.exact(x[0],x[1],t)[0],Z.sub(1)))
    else:
        u0.assign(zinit.sub(0))
        rho0.assign(zinit.sub(1))

    #plot(u0)
    tripcolor(rho0)
    plt.show()
    #Define timestep
    dt = prob.dt
    
    #Build weak form
    phi, psi = TestFunctions(Z)
    z1 = Function(Z)
    z_trial = TrialFunction(Z)
    u_trial, rho_trial = split(z_trial)
    z1.assign(z0)

    u0, rho0 = split(z0)

    ut = (u_trial - u0) / dt
    rhot = (rho_trial - rho0) / dt
    umid = 0.5 * (u_trial + u0)
    rhomid = 0.5 * (rho_trial + rho0)

    F1 = inner(ut, phi) * dx \
        + prob.f * inner(as_vector((-umid[1],umid[0])) , phi) * dx \
        - prob.c**2 * rhomid * div(phi) * dx

    F2 = (rhot + div(umid)) * psi * dx
    
    F = F1 + F2

    
    #Read out A and b
    # print(assemble(lhs(F),mat_type='aij').M.handle)
    # input('')
    # A = assemble(lhs(F),mat_type='aij').M.handle.getValuesCSR()
    # A = sp.sparse.csr_matrix((A[2],A[1],A[0]))
    A = assemble(lhs(F),mat_type='aij').M.values
    b = refd.combine(assemble(rhs(F)).dat.data)

    
    # #Get form for mass matrix (has LHS and RHS)
    # M_form =  psi * dx + inner((phi+u0)/2,n)*dS ###fix this?
    # M = refd.combine(assemble(rhs(M_form)).dat.data)
    # #Mr = refd.combine(assemble(rhs(M_form)).dat.data)

    #And for L
    L_form = inner(u_trial, phi) * dx + prob.c**2*rho_trial*psi * dx
    L = assemble(lhs(L_form),mat_type='aij').M.values

    omega = refd.combine(assemble(psi * dx).dat.data)
    
    #Get the initial values for the invariants
    m0 = assemble(rho0*dx)
    e0 = assemble(0.5*inner(u0,u0)*dx + 0.5*prob.c**2*rho0**2*dx)
    

    #Generate output dictionary
    out = {
        'A': A,
        'b': b,
        # 'M': M,
        # 'Mr': Mr,
        'omega': omega,
        'L': L,
        'm0': m0,
        'e0': e0,
    }
        
    return out, prob


def compute_invariants(prob,uvec):

    #set up DG stuff
    n = FacetNormal(prob.mesh)
        
    z = refd.nptofd(prob,uvec)
    
    u,rho = z.split()
    mass = assemble(rho*dx)
    energy = assemble(0.5 * inner(u,u)*dx + 0.5*prob.c**2 * rho**2*dx)
    pv = 0 
    
    inv_dict = {'mass' : mass,
                'energy' : energy}

    return inv_dict


if __name__=="__main__":
    dict, prob = linforms()
    print(dict)

    refd.nptofd(prob,dict['b'])
