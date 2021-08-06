#global
from firedrake import *
import numpy as np
import numpy.linalg as npl
import scipy.sparse.linalg as spsla
import matplotlib.pylab as plt
import pyamg.krylov as pak
import scipy.sparse as sps
#local
import swe
import refd2 as refd
from solve import gmres
from sweLinearSolver import swesolver
import sweVisualise as vis


if __name__=="__main__":

    params, prob = swe.linforms(degree=1)

    k = 20

    #Get initial conditions
    Z = prob.function_space(prob.mesh)
    z0 = Function(Z)
    x, y = SpatialCoordinate(prob.mesh)
    z0.sub(1).assign(project(prob.exact(x,y,0.)[0],Z.sub(1)))
    
    x, solvedict = gmres(params['A'],
                         params['b'],
                         x0=np.zeros_like(params['b']),
                         k=k)
        
    x_con, geodict = swesolver(params,
                              x0=np.zeros_like(params['b']),
                              k=k)
                             

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
    invamg = swe.compute_invariants(prob,x_pak)
    print('pyamg mass deviation =', invamg['mass']-params['m0'])
    print('pyamg energy deviation =', invamg['energy']-params['e0'])

    
    #compute invariants for direct solve
    invdir = swe.compute_invariants(prob,x_dir)
    print('direct solver mass deviation =', invdir['mass']-params['m0'])
    print('direct solver energy deviation =', invdir['energy']-params['e0'])
    
    input('pause')

    if prob.dim==2:
        vis.tabulator(params,prob,[solvedict,geodict])
    else:
        vis.tabulator3(params,prob, [solvedict, geodict])
        
    # input('pause before plots')
    
    # #plot some solutions
    # Z = prob.function_space(prob.mesh())
    # x_fd = SpatialCoordinate(Z.mesh())
    # z0 = Function(Z)
    # u0,v0 = z0.split()
    # u0.assign(project(prob.exact(x_fd[0],0),Z.sub(0)))
    # plot(u0)
    # for i in range(0,k):
    #     z = refd.nptofd(prob,x[i])
    #     z2 = refd.nptofd(prob,x_con[i])
    #     plot(z.sub(0))
    #     plot(z2.sub(0))
    #     inv = lkdv.compute_invariants(prob,x[i])
    #     inv2 = lkdv.compute_invariants(prob,x_con[i])
    #     print('Standard GMRES:')
    #     print('mass dev =', inv['mass']-params['m0'])
    #     print('energy dev =', inv['energy']-params['e0'])

    #     print("'Conservative' GMRES:")
    #     print('mass dev =', inv2['mass']-params['m0'])
    #     print('energy dev =', inv2['energy']-params['e0'])
    #     plt.show()
