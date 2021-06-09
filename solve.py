#global
from firedrake import *
import numpy as np
import scipy.sparse.linalg as spsla
import matplotlib.pylab as plt
import pyamg.krylov as pak
#local
import lkdv
import refd
import geosolve as gs
import visualise as vis


class krylov_counter_gmres(object):
    def __init__(self,disp=True):
        self._disp = disp
        self.niter = 0
    def __call__(self, rk=None):
        self.niter += 1
        if self._disp:
            print('iter %3i\trelative rk = %s' % (self.niter, str(rk)))
    def num_its(self):
        return self.niter

def gmres(A, b, x0, k, M = None):

    #If not using preconditioner, set up identity as placeholder
    if M is None:
        M = np.identity(np.size(A[0,:]))

    #Check preconditioner dimensions make sense
    if np.shape(A)!=np.shape(M):
        raise ValueError('The matrix A must have the same structure',
                         ' as preconditioner M')
        
    x = []
    residual = []
    r = (b - np.dot(A,x0)) #define r0

    x.append(r)

         
    q = np.zeros((k+1,np.size(r)))

    beta = np.linalg.norm(r)
    
    q[0] = r / beta #normalise r0
    
    #convert type
    
    h = np.zeros((k+1,k))
    
    for j in range(k):
        y = np.asarray(A @ M @ q[j])
        
        for i in range(j+1):
            h[i,j] = np.dot(q[i],y)
            y = y - h[i,j] * q[i]
        h[j+1,j] = np.linalg.norm(y)
        if (h[j+1,j] != 0):
            q[j+1] =  y / h[j+1,j]

        res = np.zeros(j+2)
        res[0] = beta
        
        yk = np.linalg.lstsq(h[:j+2,:j+1], res)[0]
        
        x.append(M @ np.transpose(q[:j+1,:]) @ yk + x0)
        residual.append(np.linalg.norm(A @ x[-1] - b))

    #Build output dictionary
    dict = {'name': 'gmres',
            'x':x,
            'res':residual}

    return x, dict


if __name__=="__main__":

    params, prob = lkdv.linforms()

    k = 7
    
    x, solvedict = gmres(params['A'],
              params['b'],
              x0=np.zeros_like(params['b']),
              k=k)

    x_con, geodict = gs.gmres_e(A = params['A'], b = params['b'],
                       x0=np.zeros_like(params['b']),
                       k=k,
                       M = params['M'], L = params['L'],
                       omega = params['omega'], m0 = params['m0'],
                       e0 = params['e0'])
                             

    # x_pak, _ = pak.gmres(params['A'],
    #                      params['b'],
    #                      x0=np.zeros_like(params['b']),
    #                      maxiter=k,
    #                      tol= 1e-10,
    #                      orthog='mgs')


    print('gmres error on conservation =', np.max(np.abs(x_con[-1]-x[-1])/x[-1]))
    input('pause')

    vis.tabulator(params,prob,[solvedict,geodict])
    
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
