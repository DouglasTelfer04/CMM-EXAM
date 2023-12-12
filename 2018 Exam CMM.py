#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 20:03:10 2023

@author: douglas
"""
##QUESTION 1

import numpy as np
from scipy.optimize import minimize

cm = 0.7
cd = 0.9
# Objective function
def objective(x):
    d, t = x
    # Objective function: minimize the total cost
    return (7800 * (np.pi * ((d/2)**2 - (d/2 - t)**2)) * 2.75) * cm + d * cd


# Constraint function: total volume of the cylinder should be 0.8 cubic meters
def constraint(x):
    d, t = x
    return 20000/(np.pi * ((d/2)**2 - (d/2 - t)**2)) - 0.8 * (np.pi * 200E9 * np.pi * ((d**4 - (d - 2*t)**4)/64) /(2.75**2 * d * t))

bounds = [(0.01, 0.1), (0.001, 0.01)]
 # assuming meters for diameter and height

# Initial guess
initial_guess = [0.05, 0.005]  # starting with a diameter of 1 meter and height of 2 meters

# Solving the optimization problem
sol = minimize(objective, initial_guess, constraints={'type': 'eq', 'fun': constraint}, bounds = bounds)

# Display the solution
print(sol)

# Extracting the optimal values
optimal_Diameter, optimal_height = sol.x

# Calculate the optimal total cost using the optimal diameter and height
optimal_total_cost = objective([optimal_Diameter, optimal_height])

# Display the total cost and optimal values
print(f"Optimal Total Cost: £{optimal_total_cost:.2f}")
print(f"Optimal Diameter: {optimal_Diameter} meters")
print(f"Optimal t: {optimal_height} meters")






















##QUESTION 2

from sympy import symbols, Function, dsolve, Eq, sqrt, cosh

# Define symbols
x, T, w, y0 = symbols('x T w y0')
y = Function('y')

# Given solution
y_x = T/w * cosh(w/T * x) + y0 - T/w

# Differentiate y(x) with respect to x
dy_dx = y_x.diff(x)
d2y_dx2 = y_x.diff(x, 2)

# Substitute y(x), dy/dx, and d^2y/dx^2 into the differential equation
differential_eq = d2y_dx2 - w/T * sqrt(1 + (dy_dx)**2)

# Simplify the result
differential_eq_simplified = differential_eq.simplify()

# Display the results
print("Given solution:")
print(y_x)

print("\nFirst derivative:")
print(dy_dx)

print("\nSecond derivative:")
print(d2y_dx2)

print("\nDifferential equation:")
print(differential_eq_simplified)






import sympy as sym

# Define symbols
w, Ta, x, y0 = sym.symbols('w Ta x y0')
# Define y as a function of x
y = sym.Function('y')(x)

# Define the given solution (Equation C)
solution = (Ta/w) * sym.cosh((w/Ta)*(x)) + y0 - (Ta/w)

# Compute the first and second derivatives of the given solution
dydx = sym.diff(solution, x)
d2ydx2 = sym.diff(dydx, x)

print(d2ydx2)

# Equation B to verify
equation_B = d2ydx2 - (w/Ta) * sym.sqrt(1 + dydx**2)

# Simplify to check if Equation B is satisfied (should simplify to 0)
verification = sym.simplify(equation_B)

print(verification)

# The expression inside the square root, cosh(w*x/Ta)**2, should simplify to cosh(w*x/Ta) itself
# This is because cosh(x) is always positive, and thus the square root of cosh(x)^2 is cosh(x)

# Let's manually simplify the verification step, considering this property of the hyperbolic cosine
manual_verification = -w * (sym.cosh(w*x/Ta) - sym.cosh(w*x/Ta)) / Ta

# Simplify the manual verification expression
manual_simplified = sym.simplify(manual_verification)
print(manual_simplified)


















x = 50
y = 15
y0 = 5
w = 10


'''
def newton(f,Df,x0,epsilon,max_iter):
    
    xn = x0
    for n in range(0,max_iter):
        fxn = f(xn)
        if abs(fxn) < epsilon:
            print('Found solution after',n,'iterations.')
            return xn
        Dfxn = Df(xn)
        if Dfxn == 0:
            print('Zero derivative. No solution found.')
            return None
        xn = xn - fxn/Dfxn
    print('Exceeded maximum iterations. No solution found.')
    return None

f = lambda x: T/w * np.cosh(w/T * x) + y0 - T/w -y
df= lambda x: np.sinh(w * x/T)
x0=1000
epsilon=0.001
max_iter=100
solution = newton(f,df,x0,epsilon,max_iter)
print(solution)

'''


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
f = lambda T: T/w * np.cosh(w/T * x) + y0 - T/w -y
approx_T = bisection(f,1100,1300,100)
print(approx_T)

print(f'answer is demonstratably true as when subbed into equation zero achieved{approx_T/w * np.cosh(w/approx_T * x) + y0 - approx_T/w -y}')




##Fixed point iteration


G = 20
D = 500



def newton_raphson(e, tolerance, max_iterations):
    root = initial_guess
    num_iterations = 0

    while num_iterations < max_iterations:
        # Calculate the function value and its derivative
        f = G * e**3 / (1 - e) - 150 * (1 - e) / D - 1.75
        f_prime = (G * (e**4 - 2 * e**3 + 3 * e**2)) / (1 - e)**2 + 150 / D

        # Update the root using the Newton-Raphson formula
        new_root = e - f / f_prime

        # Check for convergence
        if abs(new_root - e) < tolerance:
            return new_root, num_iterations + 1

        e = new_root
        num_iterations += 1

    return e, num_iterations

# Example usage:
initial_guess = 0.5
tolerance = 1e-6
max_iterations = 1000

root, num_iterations = newton_raphson(initial_guess, tolerance, max_iterations)

print(f"Approximate root: {root}")
print(f"Number of iterations: {num_iterations}")






















##QUESTION 3





print("")
import numpy as np

def deflate_polynomial(coefficients, root):
    """
    Deflate a polynomial by dividing it by a root using synthetic division.

    Parameters:
    - coefficients: Coefficients of the polynomial in descending order.
    - root: The root to be used for deflation.

    Returns:
    - deflated_coefficients: Coefficients of the deflated polynomial.
    """
    n = len(coefficients) - 1  # Degree of the original polynomial

    # Initialize the deflated coefficients with zeros
    deflated_coefficients = np.zeros(n)

    # Perform synthetic division
    deflated_coefficients[0] = coefficients[0]
    for i in range(1, n):
        deflated_coefficients[i] = root * deflated_coefficients[i-1] + coefficients[i]

    return deflated_coefficients

# Given coefficients for the 4th-order polynomial
coefficients = [1, 5, 15, 3, -10]

# Initialize a set to store the roots
roots = set()

# Perform polynomial deflation to find all roots
while len(coefficients) > 1:
    # Find a root using NumPy's roots function
    root = np.roots(coefficients)[-1]

    # Check if the root is complex
    if np.iscomplexobj(root):
        # Append the root and its conjugate to the set
        roots.add(root)
        roots.add(np.conj(root))
    elif np.abs(root.imag) > 1e-10:
        # If the imaginary part is not negligible, consider it as part of a complex conjugate pair
        roots.add(root)
        roots.add(np.conj(root))
    else:
        # Append the real root to the set
        roots.add(root.real)

    # Deflate the polynomial
    coefficients = deflate_polynomial(coefficients, root)

print("All deflated roots:", roots)





##above method gives one false root


import numpy as np

def evaluate_polynomial(coefficients, x):
    """
    Evaluate a polynomial defined by coefficients at the given value(s) x.

    Parameters:
    - coefficients: Coefficients of the polynomial in descending order.
    - x: Value or array of values at which to evaluate the polynomial.

    Returns:
    - value: Result of the polynomial evaluation.
    """
    return np.polyval(coefficients, x)

# Example usage:
# Given coefficients for the 4th-order polynomial
coefficients = [1, 5, 15, 3, -10]

# Substitute a single value (complex or real) into the polynomial
x_single_value = 1 # Replace with any complex or real value
result_single_value = evaluate_polynomial(coefficients, x_single_value)
print(f"The polynomial value at {x_single_value} is: {result_single_value}")

# Substitute an array of values (complex or real) into the polynomial
x_array_values = np.array([0.6553476519035827+0j,-2.267814020474939+2.9128348232379464j ,-2.267814020474939-2.9128348232379464j ,-2.267814020474939, -1.1197196109537042+0j ])  # Replace with any array of complex or real values
result_array_values = evaluate_polynomial(coefficients, x_array_values)
print(f"The polynomial values at {x_array_values} are: {result_array_values}")


