#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 19:00:05 2023

@author: douglas
"""
##Trying to maximise revenue in engineering project
##minimize method

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
import matplotlib.cm as cm

# Objective function
def objective(x):
    D, L = x
    # Objective function: minimize the total cost
    return -160 * D**(2/3) * L**(1/3)


# Constraint function: total volume of the cylinder should be 0.8 cubic meters
def constraint(x):
    D, L = x
    return 0.15 * L + 20 * D - 20000

# Bounds for the variables (diameter and height)
 # assuming meters for diameter and height

# Initial guess
initial_guess = [1, 2]  # starting with a diameter of 1 meter and height of 2 meters

# Solving the optimization problem
sol = minimize(objective, initial_guess, constraints={'type': 'eq', 'fun': constraint})

# Display the solution
print(sol)

# Extracting the optimal values
optimal_Diameter, optimal_height = sol.x

# Calculate the optimal total cost using the optimal diameter and height
optimal_total_cost = objective([optimal_Diameter, optimal_height])

# Display the total cost and optimal values
print(f"Optimal Total Cost: ${optimal_total_cost:.2f}")
print(f"Optimal Diameter: {optimal_Diameter} meters")
print(f"Optimal Height: {optimal_height} meters")













##Lagrange method


def objective(X):
    x, y = X
    return (160*x**0.66)*(y**0.33)
#This is the constraint function that has lambda as a coefficient.
def eq(X):
    x, y = X
    return 20*x + 0.15*y - 20000.
import autograd.numpy as np
from autograd import grad
def F(L):
    'Augmented Lagrange function'
    x, y, _lambda = L
    return objective([x, y]) + _lambda *eq([x, y])
# Gradients of the Lagrange function
dfdL = grad(F, 0)

# Find L that returns all zeros in this function.
def obj(L):
    x, y, _lambda = L
    dFdx, dFdy, dFdlam = dfdL(L)
    return [dFdx, dFdy, eq([x, y])]
from scipy.optimize import fsolve
x, y, _lam = fsolve(obj, [1., 1.,
1.0])
print(f'The answer is at {x, y}')

###The provided code is implementing a numerical optimization method known as the Augmented Lagrangian
 #method. This approach is used for solving constrained optimization problems, where both equality and
 #inequality constraints are present. The Augmented Lagrangian method introduces penalty terms into the
 #objective function to enforce the constraints and then iteratively updates Lagrange multipliers to 
# converge to a solution.

















##Tutorial 11



import numpy as np

def gsection(ftn, xl, xm, xr, tol = 1e-9):
    # applies the golden-section algorithm to maximise ftn
    # we assume that ftn is a function of a single variable
    # and that x.l < x.m < x.r and ftn(x.l), ftn(x.r) <= ftn(x.m)
    
    # In this context, xm is an initial guess x-value for the optimum point
    #
    # The algorithm iteratively refines x.l, x.r, and x.m and
    # terminates when x.r - x.l <= tol
    
    # To convert to a minimization code, you need to switch
    # the sign of the inequality tests in the if statements on lines 24 and 38.
    
    
    gr1 = 1 + (1 + np.sqrt(5))/2
    #
    # successively refine x.l, x.r, and x.m
    fl = ftn(xl)
    fr = ftn(xr)
    fm = ftn(xm)
    while ((xr - xl) > tol):
        if ((xr - xm) > (xm - xl)):
            y = xm + (xr - xm)/gr1
            fy = ftn(y)
            if (fy >= fm):
                xl = xm
                fl = fm
                xm = y
                fm = fy
            else:
                xr = y
                fr = fy
        else:
            y = xm - (xm - xl)/gr1
            fy = ftn(y)
            if (fy >= fm):
                xr = xm
                fr = fm
                xm = y
                fm = fy
            else:
                xl = y
                fl = fy     
    return(xm, ftn(xm))
    
xl=0
xm=2
xr=10
def ftn(x):
    return 2*np.sin(x)-(x**2/10)
print(gsection(ftn, xl, xm, xr, tol = 1e-9))




#The golden ratio, denoted by gr1, is used to successively refine the interval. The golden ratio ensures
# that the interval is split in a way that avoids redundant function evaluations.
#Successive Refinement: The algorithm iteratively refines the interval by evaluating the function at
# two new points (y) derived from the current interval and the golden ratio. It compares the function 
#values at the new points with the midpoint of the interval.
#Interval Update: Depending on the comparison of function values, the interval is updated by discarding
# one of the subintervals. This process continues until the width of the interval (xr - xl) becomes
# smaller than the specified tolerance (tol).
#Optimum Point: The algorithm returns the midpoint (xm) of the final interval as the estimated optimum
# point for maximizing the function. Additionally, it provides the corresponding function value at this
# optimum point.
#Limitation is may converge to a local maximum













import sympy
import numpy as np
x,y = sympy.symbols('x,y')

#need the following to create functions out of symbolix expressions
from sympy.utilities.lambdify import lambdify
from sympy import symbols, Matrix, Function, simplify, exp, hessian, solve, init_printing
init_printing()

ka=9.
kb=2.
La=10.
Lb=10.
F1=2.
F2=4.

X = Matrix([x,y])

f = Matrix([0.5*(ka*((x**2+(La-y)**2)**0.5 - La)**2)+0.5*(kb*((x**2+(Lb+y)**2)**0.5 - Lb)**2)-F1*x-F2*y])
print(np.shape(f))

#Since the Hessian is 2x2, then the Jacobian should be 2x1 (for the matrix multiplication)
gradf = simplify(f.jacobian(X)).transpose()
# #Create function that will take the values of x, y and return a jacobian
# #matrix with values
fgradf = lambdify([x,y], gradf)
print('Jacobian f', gradf)

hessianf = simplify(hessian(f, X))
# #Create a function that will return a Jessian matrix with values
fhessianf = lambdify([x,y], hessianf)
print('Hessian f', hessianf)


def Newton_Raphson_Optimize(Grad, Hess, x,y, epsilon=0.000001, nMax = 200):
    #Initialization
    i = 0
    iter_x, iter_y, iter_count = np.empty(0),np.empty(0), np.empty(0)
    error = 10
    X = np.array([x,y])

    #Looping as long as error is greater than epsilon
    while np.linalg.norm(error) > epsilon and i < nMax:
        i +=1
        iter_x = np.append(iter_x,x)
        iter_y = np.append(iter_y,y)
        iter_count = np.append(iter_count ,i)
        print(X)

        X_prev = X
        #X had dimensions (2,) while the 2nd term (2,1), so it had to be converted to 1D
        X = X - np.matmul(np.linalg.inv(Hess(x,y)), Grad(x,y)).flatten()
        error = X - X_prev
        x,y = X[0], X[1]

    return X, iter_x,iter_y, iter_count

root,iter_x,iter_y, iter_count = Newton_Raphson_Optimize(fgradf,fhessianf,1,1)
print(root)



print(0.5*(ka*((iter_x**2+(La-iter_y)**2)**0.5 - La)**2)+0.5*(kb*((iter_x**2+(Lb+iter_y)**2)**0.5 - Lb)**2)-F1*iter_x-F2*iter_y)












