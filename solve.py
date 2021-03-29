#global
from firedrake import *
import numpy as np
import scipy.sparse.linalg as spsla
import matplotlib.pylab as plt
import pyamg.krylov as pak
#local
import lkdv
import refd


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

def gmres(A, b, x0, k):

    x = []
    r = (b - np.dot(A,x0)) #define r0

    x.append(r)

         
    q = np.zeros((k+1,np.size(r)))

    beta = np.linalg.norm(r)
    
    q[0] = r / beta #normalise r0
    
    #convert type
    
    h = np.zeros((k+1,k))
    
    for j in range(k):
        y = np.asarray(A @ q[j])
        
        for i in range(j):
            h[i,j] = np.dot(q[i],y)
            y = y - h[i,j] * q[i]
        h[j+1,j] = np.linalg.norm(y)
        if (h[j+1,j] != 0):
            q[j+1] =  y / h[j+1,j]

        res = np.zeros(j+2)
        res[0] = beta

        
        yk = np.linalg.lstsq(h[:j+2,:j+1], res)[0]
        # print(yk)

        # print(np.shape(q))
        # print(np.shape(yk))
        # print(np.shape(x0))
        # input('pause')
        
        x.append(np.transpose(q[:j+1,:]) @ yk + x0)
        # print(np.shape(x))
        # input('pause')
    
    
    print(x)
    


    return x



if __name__=="__main__":

    params, prob = lkdv.linforms()

    k = 10
    
    x = gmres(params['A'],
              params['b'],
              x0=np.zeros_like(params['b']),
              k=k)

    counter = krylov_counter_gmres()
    
    x_benchmark, _ = spsla.gmres(params['A'],
                                 params['b'],
                                 x0=np.zeros_like(params['b']),
                                 tol=1e-1000,
                                 maxiter=k,
                                 restart=2*k,
                              callback=counter)

    x_pak, _ = pak.gmres(params['A'],
                         params['b'],
                         x0=np.zeros_like(params['b']),
                         maxiter=k,
                         tol= 1e-10)

    print(np.shape(x[-1]))
    print(np.shape(x_pak))
    input('pause')
    print(x_pak-x[-1])
    input('pause')

    #plot some solutions
    Z = prob.function_space(prob.mesh())
    x_fd = SpatialCoordinate(Z.mesh())
    z0 = Function(Z)
    u0,v0 = z0.split()
    u0.assign(project(prob.exact(x_fd[0],0),Z.sub(0)))
    plot(u0)
    for i in range(k-1,k):
        z = refd.nptofd(prob,x[i])
        z2 = refd.nptofd(prob,x_pak)
        zd = refd.nptofd(prob,x[i]-x_pak[:])
        plot(z.sub(1))
        plot(z2.sub(1))
        plot(zd.sub(1))
        
        plt.show()
