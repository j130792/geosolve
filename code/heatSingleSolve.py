#global
from firedrake import *
import numpy as np
import numpy.linalg as npl
import scipy.sparse.linalg as spsla
import matplotlib.pylab as plt
import pyamg.krylov as pak
import scipy.sparse as sps
#local
import heat
import refds as refd
from solve import gmres
from heatLinearSolver import heatsolver
import heatVisualise as vis

import random

if __name__=="__main__":

    params, prob = heat.linforms(degree=2)

    k = 20

    #Define preconditioner
    M = spsla.spilu(params['A'], drop_tol=1e-3,
                    fill_factor = 10)

    
    #Get initial conditions
    Z = prob.function_space(prob.mesh)
    z0 = Function(Z)
    x, y = SpatialCoordinate(prob.mesh)
    z0.assign(project(prob.exact(x,y,0.),Z))
    # for w in range(len(z0.dat.data)):
    #     z0.dat.data[w] = random.random()
    
    x, solvedict = gmres(params['A'],
                         params['b'],
                         x0=np.zeros_like(params['b']),
                         k=k,
                         pre=M)
    #Append old value of z to dictionary
    solvedict['z0'] = z0
        
    x_con, geodict = heatsolver(params,
                                x0=np.zeros_like(params['b']),
                                k=k,
                                pre=M)
    #Append old value of z to dictionary
    geodict['z0'] = z0

    x_pak, _ = pak.gmres(params['A'],
                         params['b'],
                         x0=np.zeros_like(params['b']),
                         maxiter=k,
                         tol= 1e-10,
                         orthog='mgs')

    x_dir = spsla.spsolve(params['A'],params['b'])


    print('gmres error on conservation =', np.max(np.abs(x_con-x)/x))
    print('gmres error on standard =', np.max(np.abs(x_pak-x)/x))

    #compute invariants for pyamg solve
    invamg = heat.compute_invariants(prob,x_pak,z0)
    print('pyamg mass deviation =', invamg['mass']-params['m0'])
    print('pyamg energy deviation =', invamg['energy']-params['e0'])

    
    #compute invariants for direct solve
    invdir = heat.compute_invariants(prob,x_dir,z0)
    print('direct solver mass deviation =', invdir['mass']-params['m0'])
    print('direct solver energy deviation =', invdir['energy']-params['e0'])
    
    input('pause')
    
    vis.tabulator(params,prob,[solvedict,geodict])
        
