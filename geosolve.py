"""
Here we construct modified gmres algorithms
"""
#Global imports
import numpy as np
import matplotlib.pylab as plt
import scipy.optimize as spo

def gmres_e(A, b, x0, k,
            M, L, omega, m0, e0,
            pre = None):

    #If not using preconditioner, set up identity as placeholder
    if pre is None:
        pre = np.identity(np.size(A[0,:]))

    #Check preconditioner dimensions make sense
    if np.shape(A)!=np.shape(pre):
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
        y = np.asarray(A @ pre @ q[j])
        
        for i in range(j+1):
            h[i,j] = np.dot(q[i],y)
            y = y - h[i,j] * q[i]
        h[j+1,j] = np.linalg.norm(y)
        if (h[j+1,j] != 0):
            q[j+1] =  y / h[j+1,j]

        res = np.zeros(j+2)
        res[0] = beta

        Q = np.transpose(q[:j+1,:]) #Allocate current size of Q
        Qt = np.transpose(Q)

        #Set up function
        def func(z):
            F = res - h[:j+2,:j+1] @ z
            out = np.inner(F,F)
            return out

        #Add first constraint
        def const1(z):
            X = x0 + Q @ z
            out = np.transpose(omega) @ X - m0
            return out
        con1 = {"type": "eq",
                "fun": const1}

        #second constraint
        def const2(z):
            X = x0 + Q @ z
            out = 0.5 * np.transpose(X) @ L @ X \
                - 0.5 * np.transpose(X) @ M @ X \
                - e0
            return out
        con2 = {"type": "eq",
                "fun": const2}

        eps = 1e-14
        #Initialise guess
        y0 = np.zeros((j+1,))
        if j==0:
            yk = y0
        y0[:-1] = yk
        #For the first iteration just use gmres
        if 1>0:#j==0:
            yk = spo.minimize(func,y0,tol=eps,
                              method='SLSQP',
                              options={'gtol': eps,
                                       'xtol': eps,
                                       'barrier_tol': eps,}).x
        #     #Second iteration add mass constraint
        # elif j==1:
        #     y0[:-1] = yk
        #     yk = spo.minimize(func,y0,tol=eps,
        #                       constraints=[],
        #                       method='trust-constr',
        #                       options={'gtol': eps,
        #                                'xtol': eps,
        #                                'barrier_tol': eps}).x
        #     #For all other iterations add both constraints
        # else:
        #     y0[:-1] = yk
        #     yk = spo.minimize(func,y0,tol=eps,
        #                       constraints=[],
        #                       method='trust-constr',
        #                       options={'gtol': eps,
        #                                'xtol': eps,
        #                                'barrier_tol': eps}).x
        
        x.append(pre @ Q @ yk + x0)

        #Compute residual
        residual.append(np.linalg.norm(A @ x[-1] - b))


    #build output dictionary
    dict = {'name':'geosolve',
            'x': x,
            'res': residual}

    return x, dict

