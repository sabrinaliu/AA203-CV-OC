import numpy as np
import cvxpy as cp
from scipy.optimize import minimize, Bounds

import scipy as sp

# fixed random seed
np.random.seed(0)

#define constants
#do these bounds vary from patient to patient?
#note: x0 = pau, x1 = pal, x2 = pvu, x3 = pvl, u0 = Raup, u1 = Q
x0_lb = 50
x0_ub = 100
x1_lb = 50
x1_ub = 115
x2_lb = 0.3
x2_ub = 3.5
x3_lb = 1
x3_ub = 25
u0_lb = 0.2
u0_ub = 1.1
u1_lb = 0
u1_ub = 200
x_init = np.array([(x0_lb+x0_ub)/2, (x1_lb+x1_ub)/2, (x2_lb+x2_ub)/2, (x3_lb+x3_ub)/2])
# x_init = np.zeros(4)
# RCVals = (1.7, 0.26, 51, 4.3, 0.15, 7.1, 0.028) #does this vary from patient to patient?
# patientNum = 12726

def f(xi, ui, RCVals):
    (Cau, Cal, Cvu, Cvl, Ral, Ralp, Rvl) = RCVals

    # intermediate variables: flows
    # qaup = (xi[0] - xi[2]) * cp.inv_pos(ui[0]) # AHHH FUDGE THIS MAY BE NONCONVEX... ohhh that's why supposed to use scipy minimize oops yea rewrite all this
    qaup = (xi[0] - xi[2]) / ui[0]
    qal = (xi[0] - xi[1]) / Ral
    qalp = (xi[1] - xi[3]) / Ralp
    qvl = (xi[3] - xi[2]) / Rvl

    return [(ui[1] - qal - qaup)/ Cau, (qal - qalp)/ Cal, (qaup - qvl)/ Cvu, (qalp - qvl)/ Cvl]

def solveNOC(paud, RCVals, Qd, h):
    N = paud.size - 1

    #Decision variables
    #z = np.concatenate([x1, x2, x3, x4, u1, u2])
    #note: x1 = pau, x2 = pal, x3 = pvu, x4 = pvl, u1 = Raup, u2 = Q
    get_x0 = lambda z: z[:N+1]
    get_x1 = lambda z: z[N+1:2*(N+1)]
    get_x2 = lambda z: z[2*(N+1):3*(N+1)]
    get_x3 = lambda z: z[3*(N+1):4*(N+1)]
    get_u0 = lambda z: z[4*(N+1):4*(N+1)+N]
    get_u1 = lambda z: z[4*(N+1)+N:4*(N+1)+2*N]

    #set up problem and 'minimize'
    eps = 1e-3
    none_arr = np.empty(N+1)
    none_arr[:] = None
    bounds = Bounds(
        # np.concatenate([None, None, None, None, u0_lb*np.ones(N), u1_lb*np.ones(N)]),
        # np.concatenate([None, None, None, None, u0_ub*np.ones(N), u1_ub*np.ones(N)])
        np.concatenate([x0_lb*np.ones(N+1), x1_lb*np.ones(N+1), x2_lb*np.ones(N+1), x3_lb*np.ones(N+1), u0_lb*np.ones(N), u1_lb*np.ones(N)]),
        np.concatenate([x0_ub*np.ones(N+1), x1_ub*np.ones(N+1), x2_ub*np.ones(N+1), x3_ub*np.ones(N+1), u0_ub*np.ones(N), u1_ub*np.ones(N)])
    )

    cost = lambda z: np.sum(np.square((get_x0(z)[:]-paud[:])/paud[:]))+ np.sum(np.square((get_u1(z)[:]-Qd)/Qd)) + 0.01 * np.sum(np.square(get_u0(z)[:]))

    def constraints(z):
        f_evaluated = f(np.array([get_x0(z)[:-1], get_x1(z)[:-1], get_x2(z)[:-1], get_x3(z)[:-1]]), np.array([get_u0(z), get_u1(z)]), RCVals)
        return np.concatenate([
            #dynamics
            get_x0(z)[1:] - get_x0(z)[:-1] - h[:-1]*(f_evaluated[0]),
            get_x1(z)[1:] - get_x1(z)[:-1] - h[:-1]*(f_evaluated[1]),
            get_x2(z)[1:] - get_x2(z)[:-1] - h[:-1]*(f_evaluated[2]),
            get_x3(z)[1:] - get_x3(z)[:-1] - h[:-1]*(f_evaluated[3]),

            #initial conditions (?)
            # get_x0(z)[0:1] - x_init[0],
            # get_x1(z)[0:1] - x_init[1],
            # get_x2(z)[0:1] - x_init[2],
            # get_x3(z)[0:1] - x_init[3],
        ])
    
    #initial guess for iteration
    z0 = np.concatenate([np.zeros(N+1)+68.0, np.zeros(N+1)+67.0, np.zeros(N+1)+3.5, np.zeros(N+1)+3.75, np.zeros(N)+0.81, np.zeros(N)+90.0])
    result = minimize(cost, 
                      z0, 
                      bounds = bounds, 
                      constraints={
                          'type': 'eq',
                          'fun': constraints
                      },
                      options={
                          'maxiter': 100,
                          'disp': True
                      }
                    )
    
    verbose = True
    if verbose:
        print(result)

    return get_x0(result.x), get_x1(result.x), get_x2(result.x), get_x3(result.x), get_u0(result.x), get_u1(result.x)

