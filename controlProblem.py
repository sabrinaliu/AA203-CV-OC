import numpy as np
import cvxpy as cp
from scipy.optimize import minimize, Bounds

import scipy as sp

# fixed random seed
np.random.seed(0)

#define constants
#note: x0 = pau, x1 = pal, x2 = pvu, x3 = pvl, u0 = Raup, u1 = Q
#note: based on bounds from the paper

x_init_steadyState = np.array([71.1, 69.7, 2.6, 2.9])
x_init_tilt = np.array([67.8, 66.4, 2.6, 2.9])

x_lb_steadyState = np.array([68.0, 68.0, 2.0, 2.7])
x_ub_steadyState = np.array([80.0, 80.0, 3.0, 3.0])

x_lb_tilt = np.array([50.0, 50.0, 0.3, 1.0])
x_ub_tilt = np.array([100.0, 115.0, 3.5, 25.0])

u0_lb_steadyState = 0.75; u0_ub_steadyState = 0.95 # only applied to start and end
u0_lb_tilt = 0.2; u0_ub_tilt = 1.1 # only applied to start and end
u1_lb = 50; u1_ub = 200 # we choose ourself

# x0_lb = 50
# x0_ub = 100
# x1_lb = 50
# x1_ub = 115
# x2_lb = 0.3
# x2_ub = 3.5
# x3_lb = 1
# x3_ub = 25
# u0_lb = 0.2
# u0_ub = 1.1
# u1_lb = 0
# u1_ub = 200
# x_init = np.array([(x0_lb+x0_ub)/2, (x1_lb+x1_ub)/2, (x2_lb+x2_ub)/2, (x3_lb+x3_ub)/2])
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

def solveNOC(paud, RCVals, Qd, h, tiltStartIdx, tiltEndIdx):
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

    x0_lb_array = np.zeros(N+1)
    x0_lb_array[:tiltStartIdx] = x_lb_steadyState[0]; #x0_lb_array[tiltStartIdx:tiltEndIdx+1] = x_lb_tilt[0]; x0_lb_array[tiltEndIdx+1:] = x_lb_steadyState[0]
    x0_lb_array[tiltStartIdx:] = x_lb_tilt[0]

    x0_ub_array = np.zeros(N+1)
    x0_ub_array[:tiltStartIdx] = x_ub_steadyState[0]; #x0_ub_array[tiltStartIdx:tiltEndIdx+1] = x_ub_tilt[0]; x0_ub_array[tiltEndIdx+1:] = x_ub_steadyState[0]
    x0_ub_array[tiltStartIdx:] = x_ub_tilt[0]

    x1_lb_array = np.zeros(N+1)
    x1_lb_array[:tiltStartIdx] = x_lb_steadyState[1]; #x1_lb_array[tiltStartIdx:tiltEndIdx+1] = x_lb_tilt[1]; x1_lb_array[tiltEndIdx+1:] = x_lb_steadyState[1]
    x1_lb_array[tiltStartIdx:] = x_lb_tilt[1]

    x1_ub_array = np.zeros(N+1)
    x1_ub_array[:tiltStartIdx] = x_ub_steadyState[1]; #x1_ub_array[tiltStartIdx:tiltEndIdx+1] = x_ub_tilt[1]; x1_ub_array[tiltEndIdx+1:] = x_ub_steadyState[1]
    x1_ub_array[tiltStartIdx:] = x_ub_tilt[1]

    x2_lb_array = np.zeros(N+1)
    x2_lb_array[:tiltStartIdx] = x_lb_steadyState[2]; #x2_lb_array[tiltStartIdx:tiltEndIdx+1] = x_lb_tilt[2]; x2_lb_array[tiltEndIdx+1:] = x_lb_steadyState[2]
    x2_lb_array[tiltStartIdx:] = x_lb_tilt[2]

    x2_ub_array = np.zeros(N+1)
    x2_ub_array[:tiltStartIdx] = x_ub_steadyState[2]; #x2_ub_array[tiltStartIdx:tiltEndIdx+1] = x_ub_tilt[2]; x2_ub_array[tiltEndIdx+1:] = x_ub_steadyState[2]
    x2_ub_array[tiltStartIdx:] = x_ub_tilt[2]

    x3_lb_array = np.zeros(N+1)
    x3_lb_array[:tiltStartIdx] = x_lb_steadyState[3]; #x3_lb_array[tiltStartIdx:tiltEndIdx+1] = x_lb_tilt[3]; x3_lb_array[tiltEndIdx+1:] = x_lb_steadyState[3]
    x3_lb_array[tiltStartIdx:] = x_lb_tilt[3]

    x3_ub_array = np.zeros(N+1)
    x3_ub_array[:tiltStartIdx] = x_ub_steadyState[3]; #x3_ub_array[tiltStartIdx:tiltEndIdx+1] = x_ub_tilt[3]; x3_ub_array[tiltEndIdx+1:] = x_ub_steadyState[3]
    x3_ub_array[tiltStartIdx:] = x_ub_tilt[3]
        
    u0_lb_array = np.empty(N); u0_lb_array[:] = None
    u0_lb_array[0] = u0_lb_steadyState; u0_lb_array[tiltStartIdx-1] = u0_lb_steadyState
    u0_lb_array[tiltStartIdx] = u0_lb_tilt; u0_lb_array[-1] = u0_lb_tilt
    # u0_lb_array[tiltStartIdx] = u0_lb_tilt; u0_lb_array[tiltEndIdx] = u0_lb_tilt
    # u0_lb_array[tiltEndIdx+1] = u0_lb_steadyState; u0_lb_array[-1] = u0_lb_steadyState

    u0_ub_array = np.empty(N); u0_ub_array[:] = None
    u0_ub_array[0] = u0_ub_steadyState; u0_ub_array[tiltStartIdx-1] = u0_ub_steadyState
    u0_ub_array[tiltStartIdx] = u0_ub_tilt; u0_ub_array[-1] = u0_ub_tilt
    # u0_ub_array[tiltStartIdx] = u0_ub_tilt; u0_ub_array[tiltEndIdx] = u0_ub_tilt
    # u0_ub_array[tiltEndIdx+1] = u0_ub_steadyState; u0_ub_array[-1] = u0_ub_steadyState

    u1_lb_array = np.empty(N); u1_lb_array[:] = None
    u1_lb_array[0] = u1_lb; u1_lb_array[tiltStartIdx-1] = u1_lb
    u1_lb_array[tiltStartIdx] = u1_lb; u1_lb_array[-1] = u1_lb
    # u1_lb_array[tiltStartIdx] = u1_lb; u1_lb_array[tiltEndIdx] = u1_lb
    # u1_lb_array[tiltEndIdx+1] = u1_lb; u1_lb_array[-1] = u1_lb

    u1_ub_array = np.empty(N); u1_ub_array[:] = None
    u1_ub_array[0] = u1_ub; u1_ub_array[tiltStartIdx-1] = u1_ub
    u1_ub_array[tiltStartIdx] = u1_ub; u1_ub_array[-1] = u1_ub
    # u1_ub_array[tiltStartIdx] = u1_ub; u1_ub_array[tiltEndIdx] = u1_ub
    # u1_ub_array[tiltEndIdx+1] = u1_ub; u1_ub_array[-1] = u1_ub

    bounds = Bounds(
        # initial steady state bounds
        np.concatenate([x0_lb_array, x1_lb_array, x2_lb_array, x3_lb_array, u0_lb_array, u1_lb_array]),
        np.concatenate([x0_ub_array, x1_ub_array, x2_ub_array, x3_ub_array, u0_ub_array, u1_ub_array])
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

            # steady state initial conditions
            get_x0(z)[0:1] - x_init_steadyState[0],
            get_x1(z)[0:1] - x_init_steadyState[1],
            get_x2(z)[0:1] - x_init_steadyState[2],
            get_x3(z)[0:1] - x_init_steadyState[3],

            # tilt start initial conditions
            get_x0(z)[tiltStartIdx:tiltStartIdx+1] - x_init_tilt[0],
            get_x1(z)[tiltStartIdx:tiltStartIdx+1] - x_init_tilt[1],
            get_x2(z)[tiltStartIdx:tiltStartIdx+1] - x_init_tilt[2],
            get_x3(z)[tiltStartIdx:tiltStartIdx+1] - x_init_tilt[3],

            # # tilt end (new steady state) conditions (but when does it really return to "steady state"?)
            # get_x0(z)[tiltEndIdx:tiltEndIdx+1] - x_init_steadyState[0],
            # get_x1(z)[tiltEndIdx:tiltEndIdx+1] - x_init_steadyState[1],
            # get_x2(z)[tiltEndIdx:tiltEndIdx+1] - x_init_steadyState[2],
            # get_x3(z)[tiltEndIdx:tiltEndIdx+1] - x_init_steadyState[3],

        ])
    
    #initial guess for iteration
    z0 = np.concatenate([np.zeros(N+1)+x_init_steadyState[0], np.zeros(N+1)+x_init_steadyState[1], np.zeros(N+1)+x_init_steadyState[2], np.zeros(N+1)+x_init_steadyState[3], np.zeros(N)+ (u0_lb_steadyState + u0_ub_steadyState)/2.0, np.zeros(N)+Qd])
    result = minimize(cost, 
                      z0, 
                      bounds = bounds, 
                      constraints={
                          'type': 'eq',
                          'fun': constraints
                      },
                      options={
                          'maxiter': 50,
                          'disp': True
                      }
                    )
    
    verbose = True
    if verbose:
        print(result)

    return get_x0(result.x), get_x1(result.x), get_x2(result.x), get_x3(result.x), get_u0(result.x), get_u1(result.x)

