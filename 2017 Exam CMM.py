#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 22:24:23 2023

@author: douglas
"""
##QUESTION 1
import numpy as np
import matplotlib.pyplot as plt
L = 800
E = 40000
I = 40000
w0 = 3.5
x = np.linspace(0, 800, 800)
y = w0/(120 * E *L) * (-x**5 + 2*L**2*x**3 - L**4*x)
plt.plot(x,y)
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

L = 800
E = 40000
I = 40000
w0 = 3.5

# Define the deflection function
def deflection(x):
    return -w0 / (120 * E * L) * (-x**5 + 2 * L**2 * x**3 - L**4 * x)

# Find the maximum deflection using optimization
result = minimize_scalar(lambda x: -deflection(x), bounds=(0, L), method='bounded')

# Extract the optimal position and deflection
optimal_position = result.x
max_deflection = -result.fun

# Print the results
print(f"Optimal Position: {optimal_position}")
print(f"Max Deflection: {max_deflection}")

# Plot the function
x = np.linspace(0, 800, 800)
y = deflection(x)
plt.plot(x, y)
plt.scatter(optimal_position, max_deflection, color='red', marker='o', label='Max Deflection')
plt.xlabel('Position (x)')
plt.ylabel('Deflection (y)')
plt.legend()
plt.show()

##TRY with golden search technique






import numpy as np

def gsection(ftn, xl, xm, xr, tol = 1e-2):
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



L = 800
E = 40000
I = 40000
w0 = 3.5    
xl=0
xm=300
xr=800
def ftn(x):
    return (-w0 / (120 * E * L) * (-x**5 + 2 * L**2 * x**3 - L**4 * x))
print(gsection(ftn, xl, xm, xr, tol = 1e-9))









import numpy as np

def gsection(ftn, xl, xm, xr, rel_tol=0.01):
    # Golden-section algorithm to maximize ftn.
    # Assumes that ftn is a function of a single variable,
    # and that xl < xm < xr, with ftn(xl), ftn(xr) <= ftn(xm).
    
    gr1 = 1 + (1 + np.sqrt(5))/2
    
    # Successively refine xl, xr, and xm
    fl = ftn(xl)
    fr = ftn(xr)
    fm = ftn(xm)
    
    while True:
        if (xr - xm) > (xm - xl):
            y = xm + (xr - xm) / gr1
            fy = ftn(y)
            if fy >= fm:
                xl = xm
                fl = fm
                xm = y
                fm = fy
            else:
                xr = y
                fr = fy
        else:
            y = xm - (xm - xl) / gr1
            fy = ftn(y)
            if fy >= fm:
                xr = xm
                fr = fm
                xm = y
                fm = fy
            else:
                xl = y
                fl = fy
        
        # Calculate relative error
        relative_error = abs((xm - xl) / xm)
        
        # Check for convergence
        if relative_error < rel_tol:
            return xm, ftn(xm)

# Parameters for the function
L = 800
E = 40000
w0 = 3.5
xl = 0
xm = 300
xr = 800

# Function to be maximized
def ftn(x):
    return (-w0 / (120 * E * L) * (-x**5 + 2 * L**2 * x**3 - L**4 * x))

# Run the golden-section algorithm with relative error as the stopping criterion
result, value = gsection(ftn, xl, xm, xr, rel_tol=0.01)
print("Optimal point:", result)
print("Function value at optimal point:", value)

#The Golden Section Search is efficient because it narrows down the search interval at a rate that
# guarantees convergence at the optimal solution. The method utilizes the golden ratio
# (approximately 1.618) to determine the next interior point, ensuring a balance between exploration 
#and exploitation.













##Q1B

import numpy as np
import matplotlib.pyplot as plt

# Function to calculate the Taylor series expansion for exp(-x) at x0 with a perturbation h
def taylor_expansion(x, x0, h, num_terms):
    series_sum = 0
    for n in range(num_terms):
        term = (-h)**n / np.math.factorial(n) * np.exp(-x0)  # Adjusted term for perturbation and initial value
        series_sum += term * (x - x0)**n
    return series_sum

# Define the true function exp(-x)
def true_function(x):
    return np.exp(-x)

# Define perturbation and initial value
h = 0.1
x0 = 1

# Generate x values for plotting
x_values = np.linspace(0, 2, 100)

# Plot the true function
plt.plot(x_values, true_function(x_values), label='True Function: $e^{-x}$')

# Plot Taylor series expansions with different numbers of terms
for num_terms in [2, 5, 10]:
    y_values = [taylor_expansion(x, x0, h, num_terms) for x in x_values]
    plt.plot(x_values, y_values, label=f'Taylor Series ({num_terms} terms)')

plt.xlabel('x')
plt.ylabel('y')
plt.title(f'Taylor Series Expansion for $e^{-x}$ with Perturbation h={h} at x0={x0}')
plt.legend()
plt.show()
















##QUESTION 2

import numpy as np
from scipy.optimize import fsolve

# Define the equation as a Python function
def equation_to_solve(x):
    return 25000 * (x * (1 + x)**6)/((1 + x)**6 - 1) - 5500

# Initial guess for the solution
initial_guess = 1

# Use fsolve to find the numerical solution
numerical_solution = fsolve(equation_to_solve, initial_guess)

# Display the numerical solution
print("Numerical Solution for x:", numerical_solution[0])






import math

def bisection(f,a,b,N):
    
    
    
    
   

    if f(a)*f(b) >= 0:
        print("Bisection method fails.")
        return None
    a_n = a
    b_n = b
    for n in range(1,N+1):
        m_n = (a_n + b_n)/2
        f_m_n = f(m_n)
        if f(a_n)*f_m_n < 0:
            a_n = a_n
            b_n = m_n
        elif f(b_n)*f_m_n < 0:
            a_n = m_n
            b_n = b_n
        elif f_m_n == 0:
            print("Found exact solution.")
            return m_n
        else:
            print("Bisection method fails.")
            return None
    return (a_n + b_n)/2

#f = lambda x: x-(1.325/(math.log((0.000005/3.7*0.005)+5.74/13,473**0.9))**2)
f = lambda x: 25000 * x * (1 + x)**6/((1 + x)**6 - 1) -5500
approx_phi = bisection(f,0.001,1,100)
print(f'interest rate found numerically is: {approx_phi}')









t = 30
u = 1.8E3
m0 = 160E3
q = 2.5E3





##initially suvat was tried but easier to just integrate velocity equation
from scipy.integrate import quad
def integrand(t, u, m0, q):
    return u * np.log(m0/(m0 - q * t))


I = quad(integrand, 0, 30, args=(u, m0, q))
print(I)







##QUESTION 3

x = np.linspace(-1000,1000,2000)
y = 30 * (1 - abs(x)/1000)**2  + 5 * (1 - abs(x)/1000)**4
plt.plot(x,y)
plt.show()




                                                      