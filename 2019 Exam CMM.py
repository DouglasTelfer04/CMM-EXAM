#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 14:27:45 2023

@author: douglas
"""

##QUESTION 2
import numpy as np
from scipy.optimize import minimize

# Objective function
def objective(P):
    return -np.cos(np.pi + P*np.pi)/(1+P) - np.cos(np.pi - P *np.pi)/(1 - P) + 1/(1+P) + 1/(1-P)

# Initial guess
initial_guess = [2]

# Solving the optimization problem
sol = minimize(objective, initial_guess)

# Display the solution
print(sol)

# Extracting the optimal values
optimal_P = sol.x

# Calculate the optimal objective value using the optimal P
minimum_objective = objective(optimal_P)

# Display the optimal objective value and corresponding P
print(f"Optimal Objective Value: {minimum_objective}")
print(f"Optimal P: {optimal_P}")


##The numerical method used by SciPy's minimize function is based on an iterative optimization
# algorithm. One of the widely used algorithms is the Sequential Least Squares Quadratic Programming
# (SLSQP) algorithm. The SLSQP algorithm is suitable for solving constrained and unconstrained nonlinear
# optimization problems.
#Gradient Descent: The algorithm uses the gradient (or numerical approximation of the gradient) of the objective function to find the steepest descent direction. It then updates the variables along this direction to minimize the function.
#Convergence Check: The algorithm checks for convergence by evaluating the change in the objective 
#function and the change in the variable values. If the changes are smaller than a specified tolerance, 
#the algorithm considers the solution converged.
#Constraint Handling: If there are constraints, the algorithm uses optimization techniques to handle
# them. It might use Lagrange multipliers or penalty functions to enforce the constraints.
#Sensitivity to Initial Guess: The performance of the algorithm can depend on the quality of the
# initial guess.
#Numerical Stability: For certain types of functions or poorly conditioned problems, the 
#numerical stability of the algorithm may become an issue.



#first plotte dto visualise where roots roughly are and then reguliFalse method used to pinpoint more accurately
##Also noted that 0 and pi are also roots


MAX_ITER = 10000
  
# An example function whose solution 
# is determined using Bisection Method.  
# The function is x^3 - x^2 + 2 
def func( x ): 
    return np.sin(x) * np.cos(x * optimal_P)
  
# Prints root of func(x) in interval [a, b] 
def regulaFalsi( a , b): 
    if func(a) * func(b) >= 0: 
        print("You have not assumed right a and b") 
        return -1
      
    c = a # Initialize result 
      
    for i in range(MAX_ITER): 
          
        # Find the point that touches x axis 
        c = (a * func(b) - b * func(a))/ (func(b) - func(a)) 
          
        # Check if the above found point is root 
        if func(c) == 0: 
            break
          
        # Decide the side to repeat the steps 
        elif func(c) * func(a) < 0: 
            b = c 
        else: 
            a = c 
    print("The value of root is : " , '%.4f' %c) 
    print(i)
  
# Driver code to test above function 
# Initial values assumed 

regulaFalsi(0.1, 1) 
regulaFalsi(2,3)























##QUESTION 4 
from scipy.optimize import fsolve
import math

def equations(p):
    e, a, c = p
    return (6870 + 6870 * e * np.sin(np.radians(-30) + a) - c, 6728 + 6728 * e * np.sin(a) - c, 6615 + 6615 * e * np.sin(np.radians(30) + a) - c)

e, a, c =  fsolve(equations, (0,0,0))

print(equations((e, a, c)))

from scipy.optimize import fsolve
from math import exp

def equations(vars):
    e, a, c = vars
    eq1 = 6870 + 6870 * e * np.sin(-30*np.pi/180 + a) - c
    eq2 = 6728 + 6728 * e * np.sin(a) - c
    eq3 = 6615 + 6615 * e * np.sin(30*np.pi/180 + a) - c
    return [eq1, eq2, eq3]

sol =  fsolve(equations, (0,0,0))

print(f'trial 1 {sol}')
def f(theta):
    
    return c/(1+e*np.sin(theta + a))

print(f(np.radians(30)))

from scipy.optimize import minimize
initial_guess = np.array([0])
result = minimize(f, initial_guess)
print(result.fun)
print(result.x)



import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import math
import numpy as np
def equations(p):
    c, e, a = p
    eqn1 = 6870 + 6870 * e*np.sin(np.radians(-30) + a) - c
    eqn2 = 6728 + 6728 * e*np.sin(a) - c
    eqn3 = 6615 + 6615 * e*np.sin(np.radians(30) + a) - c
    return (eqn1, eqn2, eqn3)
sol = fsolve(equations, (0, 0, 0))
print(f'trial 2: {sol}')


c, e, a = sol

def f(theta):
    
    return c/(1+e*np.sin(theta + a))

print(f(np.radians(74)))
print(f(np.radians(30)))

from scipy.optimize import minimize
initial_guess = np.array([0])
result = minimize(f, initial_guess)
print(result.fun)
print(result.x)










##Attempted with Newton method
print("Newton Method")
import numpy as np


# Define the system of equations
def equations(x):
    a, c, e = x
    f1 = 6870 + 6870 * e * np.sin(-np.pi/6 + a) - c
    f2 = 6728 + 6728 * e * np.sin(a) - c
    f3 = 6615 + 6615 * e * np.sin(np.pi/6 + a) - c
    return np.array([f1, f2, f3])

# Define the Jacobian matrix with regularization
def jacobian(x):
    a, _, e = x
    J = np.array([
        [6870 * e * np.cos(-np.pi/6 + a), -1, 6870 * np.sin(-np.pi/6 + a)],
        [6728 * e * np.cos(a), -1, 6728 * np.sin(a)],
        [6615 * e * np.cos(np.pi/6 + a), -1, 6615 * np.sin(np.pi/6 + a)]
    ])
    J += np.eye(3) * 1e-6  # Add a small regularization term
    return J

# Newton's method with regularization
def newton_method(x0, max_iter=500, tol=1e-4):
    x = np.array(x0, dtype=float)
    for i in range(max_iter):
        print("Iteration:", i)
        print("x before update:", x)
        dx = np.linalg.solve(jacobian(x), -equations(x))
        print("dx:", dx)
        x += dx
        print("x after update:", x)
        print("Equations:", equations(x))
        if np.linalg.norm(dx) < tol:
            print("Converged!")
            break
    else:
        print("Did not converge within maximum iterations.")
    return x

# Initial guess
initial_guess = [0, 0, 0]

# Solve using Newton's method with regularization
solution = newton_method(initial_guess)
a = solution[0]
c = solution[1]
e = solution[2]

# Display the solution
print("Solution:")
print("a =", solution[0])
print("c =", solution[1])
print("e =", solution[2])

# Check the result
def f(theta):
    return c / (1 + e * np.sin(theta + a))

print("f(74 degrees):", f(np.radians(74)))

#This code implements the Newton method for solving a system of nonlinear equations. It iteratively
 #updates the solution vector x by calculating the Newton step and checking for convergence. The
 #regularization term is added to the Jacobian to improve numerical stability. The code prints debugging
 #information for each iteration, helping to understand the convergence process. The final solution is
 #displayed along with the result of the objective function at a specific angle.

#The Jacobian matrix is a matrix of all first-order partial derivatives of a vector-valued function.
 #In the context of Newton's method for optimization, the Jacobian is used to linearize and approximate
 #the behavior of the system around the current estimate of the solution.


#newton's method effectively handles nonlinearities by iteratively refining the estimate of the solutio
#n based on the linearized system. The Jacobian allows the method to adapt its steps according to the 
#local curvature of the objective function.

#In some cases, regularization is added to the Jacobian matrix to ensure numerical stability. The 
#addition of a small regularization term (in this case, np.eye(3) * 1e-6) helps avoid issues related 
#to singular or ill-conditioned Jacobian matrices.


#You could use fsolve or another algorithm in this example alright, as it is very difficult to linearize
# in a form where you could use Gauss Elimination. Because you have three functions in three variables 
#it does not lend itself to graphical solutions (i.e. finding an intercept) as you have to adjust two 
#parameters to solve for three variables, which would be computationally intensive.


















##Question 5

print("QUESTION 5")

import numpy as np
from scipy.optimize import minimize

# Objective function
def objective(theta):
    theta_1, theta_2, theta_3 = theta
    # Objective function: minimize the total cost
    return  (-20000 * 1.2 * np.sin(theta_1) - 30000 * (1.2 * np.sin(theta_1) + 1.5 * np.sin(theta_2)))



def constraint1(theta):
    theta_1, theta_2, theta_3 = theta
    return 1.2 * np.cos(theta_1) + 1.5 * np.cos(theta_2) + 1 * np.cos(theta_3) - 3.5

def constraint2(theta):
    theta_1, theta_2, theta_3 = theta
    return 1.2 * np.sin(theta_1) + 1.5 * np.sin(theta_2) + 1 * np.sin(theta_3)



# Initial guess
initial_guess = [0.8, 0.5,0.07]  


# Solving the optimization problem
sol = minimize(objective, initial_guess, constraints=[ {'type': 'eq', 'fun': constraint1 }, {'type': 'eq', 'fun': constraint2 }]) 

# Display the solution
print(sol)

# Extracting the optimal values
optimal_theta_1, optimal_theta_2, optimal_theta_3 = sol.x

# Calculate the optimal total cost using the optimal diameter and height
Optimal_PE = abs(objective([optimal_theta_1, optimal_theta_2, optimal_theta_3]))

# Display the total cost and optimal values
print(f"Optimal PE: {Optimal_PE}")
print(f"Optimal theta1: {optimal_theta_1} rad")
print(f"Optimal theta2: {optimal_theta_2} rad")
print(f"Optimal theta3: {optimal_theta_3} rad")



##The SciPy minimize function iteratively refines the solution based on the optimization algorithm 
#and convergence criteria. The SLSQP algorithm efficiently handles problems with constraints and seeks
# to find a local minimum of the objective function while satisfying the specified constraints. Keep 
#in mind that the algorithm may find a local minimum, and the quality of the result depends on factors
# such as the choice of the initial guess and the nature of the objective function and constraints.