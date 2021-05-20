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

        res = np.zeros(2+1)
        res[0] = beta

        Q = np.transpose(q[:j+1,:]) #Allocate current size of Q
        Qt = np.transpose(Q)

        #Set up function
        def func(z):
            F = r - A @ Q @ z
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

            
        y0 = np.zeros((j+1,))
        yk = spo.minimize(func,y0,tol=1e-12,
                          constraints=[]).x

        # print(yk)
        
        # # input('pause')
            
        # # yk = np.linalg.lstsq(A @ Q, r)[0]

        # print(np.shape(r))
        # print(np.shape(res))
        # print(np.shape(yk))
        # print(np.shape(y))
        # input('')

        
        
        x.append(pre @ Q @ yk + x0)
    

    return x

