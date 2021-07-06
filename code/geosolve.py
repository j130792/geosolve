"""
Here we construct modified gmres algorithms
"""
#Global imports
import numpy as np
import matplotlib.pylab as plt
import scipy.optimize as spo
import warnings

def gmres_e(A, b, x0, k,
            M, L, omega, m0, mo0, e0,
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

        def jac(z):
            out = np.zeros_like(z)
            # #original term
            # F = r - A @ Q @ z
            # #Component wise differentiation of F
            # for j in range(len(z)):
            #     ej = np.zeros_like(z)
            #     ej[j] = 1 
            #     dF = - A @ Q @ ej

            #     #assemble j-th component of jac
            #     out[j] = 2 * np.inner(dF,F)

            #original term
            F = res - h[:j+2,:j+1] @ z
            #Component wise differentiation of F
            for m in range(len(z)):
                ej = np.zeros_like(z)
                ej[m] = 1 
                dF = - h[:j+2,:j+1] @ ej

                #assemble j-th component of jac
                out[m] = 2 * np.inner(dF,F)
                
            return out

        def hess(z):
            dim = len(z)
            out = np.zeros((dim,dim))
            for n in range(dim):
                for m in range(dim):
                    e1 = np.zeros_like(z)
                    e1[n] = 1
                    e2 = np.zeros_like(z)
                    e2[m] = 1
                    #assemble n,m-th component of hessian
                    out[n,m] = 2 * np.inner( h[:j+2,:j+1] @ e1, h[:j+2,:j+1] @ e2)

            return out
        
        #Add mass constraint
        def const1(z):
            X = x0 + Q @ z
            out = np.transpose(omega) @ X - m0
            return out
        con1 = {"type": "eq",
                "fun": const1}

        #momentum constraint
        def const2(z):
            X = x0 + Q @ z
            out = 0.5*np.transpose(X) @ M @ X - mo0
            return out
        con2 = {"type": "eq",
                "fun": const2}
        
        #energy constraint
        def const3(z):
            X = x0 + Q @ z
            out = 0.5 * np.transpose(X) @ L @ X \
                - 0.5 * np.transpose(X) @ M @ X \
                - e0
            return out
        con3 = {"type": "eq",
                "fun": const3}

        tol=1e-15
        #Initialise guess
        y0 = np.zeros((j+1,))

        # print('r', r.shape)
        # print('A', A.shape)
        # print('Q', Q.shape)
        # print('y0', y0.shape)

        # print(jac(y0))
        
        # input('p')

        #For the first iteration just use gmres
        if j==0:
            solve = spo.minimize(func,y0,tol=tol,
                              method='SLSQP',
                              options={'maxiter': 1e3,
                                       'eps': 1e-13,
                                       'iprint': 0,
                                       'ftol': 1e-50})
            if solve.message!='Optimization terminated successfully':
                warnings.warn("Iteration %d failed with message '%s'" % (j,solve.message),
                              RuntimeWarning)
            #print(solve)
            yk = solve.x

            #Second iteration add mass constraint
        elif j==1:
            y0[:-1] = yk
            solve = spo.minimize(func,y0,tol=tol,
                                 constraints=[con1],
                                 method='SLSQP',
                                 options={'maxiter': 1e3,
                                       'eps': 1e-13,
                                       'iprint': 0,
                                       'ftol': 1e-50})
            if solve.message!='Optimization terminated successfully':
                warnings.warn("Iteration %d failed with message '%s'" % (j,solve.message),
                              RuntimeWarning)
            #print(solve)
            yk = solve.x
            #Third iteration add momentum constraint
        elif j==2:
            y0[:-1] = yk
            solve = spo.minimize(func,y0,tol=tol,
                                 constraints=[con1,con3],
                                 method='SLSQP',
                                 options={'maxiter': 1e3,
                                       'eps': 1e-13,
                                       'iprint': 0,
                                       'ftol': 1e-50})
            if solve.message!='Optimization terminated successfully':
                warnings.warn("Iteration %d failed with message '%s'" % (j,solve.message),
                              RuntimeWarning)
            #print(solve)
            yk = solve.x
            #For all other iterations add both constraints
        else:
            y0[:-1] = yk
            solve = spo.minimize(func,y0,tol=tol,jac=jac,hess=hess,
                                 constraints=[con1,con2,con3],
                                 method='trust-constr',
                                 options={'xtol': 1e-25,
                                          'gtol': 1e-12,
                                          'barrier_tol': 1e-12,
                                          'maxiter': 1e3})
                                 # options={'maxiter': 1e3,
                                 #          'eps': 1e-13,
                                 #          'iprint': 0,
                                 #          'ftol': 1e-50,
                                 #          'disp': True})
            if solve.message!='Optimization terminated successfully':
                warnings.warn("Iteration %d failed with message '%s'" % (j,solve.message),
                              RuntimeWarning)
            #print(solve)
            yk = solve.x
        
        x.append(pre @ Q @ yk + x0)

        #Compute residual
        residual.append(np.linalg.norm(A @ x[-1] - b))


    #build output dictionary
    dict = {'name':'geosolve',
            'x': x,
            'res': residual}

    return x, dict

