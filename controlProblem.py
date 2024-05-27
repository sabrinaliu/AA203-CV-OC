import numpy as np
import cvxpy as cp

def f(xi, ui, RCVals):
    (Cau, Cal, Cvu, Cvl, Ral, Ralp, Rvl) = RCVals

    # intermediate variables: flows
    qaup = (xi[0] - xi[2]) * cp.inv_pos(ui[0]) # AHHH FUDGE THIS MAY BE NONCONVEX... ohhh that's why supposed to use scipy minimize oops yea rewrite all this
    qal = (xi[0] - xi[1]) / Ral
    qalp = (xi[1] - xi[3]) / Ralp
    qvl = (xi[3] - xi[2]) / Rvl

    return [(ui[1] - qal - qaup)/ Cal, (qal - qalp)/ Cal, (qaup - qvl)/ Cvu, (qalp - qvl)/ Cvl]

def solveNOC(paud, RCVals, Qd, h):
    N = paud.size - 1

    x_cvx = cp.Variable((N+1, 4)) # pau, pal, pvu, pvl
    u_cvx = cp.Variable((N, 2)) # Raup, Q

    obj = 100*((x_cvx[N,0] - paud[N]) / paud[N])**2
    constraints = []
    for i in range(N):
        obj += 100 * (((x_cvx[i,0] - paud[i]) / paud[i])**2 + ((u_cvx[i,1] - Qd)/Qd)**2) + u_cvx[i,0]**2
        constraints += [
            u_cvx[i,0] >= 0.62, u_cvx[i,0] <= 1.0,
            u_cvx[i,1] >= 81, u_cvx[i,1] <= 99,
            x_cvx[i+1] == x_cvx[i] + h * f(x_cvx[i], u_cvx[i], RCVals)
        ]
    prob = cp.Problem(cp.Minimize(obj), constraints)
    prob.solve()
    if prob.status != "optimal":
        raise RuntimeError("SCP solve failed. Problem status: " + prob.status)
    return x_cvx.value, u_cvx.value
