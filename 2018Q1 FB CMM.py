#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 15:45:10 2023

@author: douglas
"""

L = 800 # cm
E = 40000 # kN/cm^2
I = 40000 # cm^4
w0 = 3.5 # kN/cm

def bisection(f,a,b,tol):
    # Check if a and b bound a root
    if f(a)*f(b) >= 0:
       print("a and b do not bound a root")
       return None 
    a_n = a
    b_n = b
    # while 
    # how to I define my tolerance? 
    # Is my tolerance in y or in x? 
    # In x 
    # How will I define my tolerance? By a percentage change.
    result = (a_n + b_n)/2 
    previous = result - 100
    while abs((result - previous)/previous) >= tol:
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
        previous = result
        result = (a_n + b_n)/2
    return result

# Different equation 
import math
f = lambda x: ( w0 /(120 * E * I * L)) * (-5 * x**4 + 6 * L**2 * x**2 - L**4)
approx_phi = bisection(f,300,400,0.01) 
print(approx_phi)









from sympy import symbols, Eq, Function, dsolve, diff, cosh, sinh, sqrt, simplify

# Define the symbols and function again for clarity
TA, w, x, y0 = symbols('TA w x y0')
y = Function('y')(x)

# Given general solution
general_solution = (TA/w) * cosh((w/TA) * x) + y0 - (TA/w)

# First and second derivatives of the general solution
first_derivative_gen = diff(general_solution, x)
second_derivative_gen = diff(first_derivative_gen, x)

# Original differential equation
differential_eq = Eq(diff(y, x, x), (w/TA) * sqrt(1 + diff(y, x)**2))

# Substituting the derivatives of the general solution into the differential equation
differential_eq_with_gen_sol = differential_eq.subs({diff(y, x, x): second_derivative_gen, diff(y, x): first_derivative_gen})

# Simplify the differential equation with the general solution substituted in
differential_eq_simplified = simplify(differential_eq_with_gen_sol.lhs - differential_eq_with_gen_sol.rhs)

# Apply the hyperbolic identity to the simplified differential equation manually
hyperbolic_identity_applied = differential_eq_simplified.subs(cosh(w*x/TA)**2, sinh(w*x/TA)**2 + 1)

# Now simplify the equation after applying the hyperbolic identity
final_simplification = simplify(hyperbolic_identity_applied)

# Print the results
print(f"The first derivative of the general solution is: {first_derivative_gen}")
print(f"The second derivative of the general solution is: {second_derivative_gen}")
print("\nSubstituting these into the original ODE gives us:")
print(differential_eq_with_gen_sol)

print("\nAfter simplifying the substituted ODE, we get:")
print(differential_eq_simplified)

print("\nApplying the hyperbolic identity \(cosh^2(x) = sinh^2(x) + 1\) to the simplified equation:")
print(hyperbolic_identity_applied)

print("\nFinally, after applying the hyperbolic identity, the simplification is:")
print(final_simplification)

# Check if the final simplification is zero
if final_simplification == 0:
    print("\nThis confirms that the provided general solution is indeed the solution to the original ODE.")
else:
    print("\nThis does not confirm that the provided general solution is the solution to the original ODE.")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
import numpy as np
from scipy.optimize import fsolve

# Given parameters
m = 7850  # kg/m^3
L = 0.9  # m
E = 200e9  # Pa (or N/m^2)
I = 3.255e-11  # m^4

# Function defining the frequency equation
def frequency_equation(beta):
    return np.cosh(beta) * np.cos(beta) + 1

# Function to calculate the roots of the frequency equation
def find_roots():
    # Define the equation to solve
    equation_to_solve = lambda beta: frequency_equation(beta)

    # Initial guesses for roots
    initial_guesses = np.linspace(0.1, 20, 10)  # Adjust the range as needed

    # Use fsolve to find roots
    roots = fsolve(equation_to_solve, initial_guesses)

    return roots

# Convert roots to natural frequencies using the relationship: fi = beta_i^2 / (2 * pi)*(np.sqrt(m*L**3/E*I))
def calculate_frequencies(roots):
    frequencies = [beta_i**2 / (2 * np.pi)*(np.sqrt(m*L**3/E*I)) for beta_i in roots]
    return frequencies

# Find roots and corresponding natural frequencies
roots = find_roots()
frequencies = calculate_frequencies(roots)

# Print the results
print("Roots (β values):", roots)
print("Natural Frequencies (fi values):", frequencies)





import numpy
from scipy.optimize import fsolve
from matplotlib import pyplot as plt 

#initialising range of arrays to store x and y function
x = np.arange(0, np.pi, 0.01)
p = 1
y = lambda x: np.sin(x)*np.cos(p*x)

#graphing to see behaviour of function in domain
plt.plot(x, y(x), '.')

#function is zero in domain around x=0, 1.5, 3.1 which can be checked
#using scipy root solver
x1 = fsolve(y,0)
x2 = fsolve(y,1.5)
x3 = fsolve(y,3.1)

print(x1,x2,x3)

#Q2b
#values of x where function=0 in integration domain
#are at 0, 1.571, 3.141








#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                                  Q1A
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import matplotlib.pyplot as plt
import numpy as np

#Defining the equation of the beam
def f(x):

    return (w/(120*E*I*L))*(-x**5 + 2*L**2*x**3 - L**4*x)

E=40000
L=800
I=40000
w=3.5

#Plotting the function to solve the maxima though inspection

# Plotting initial function and its quadratic approximation
x = np.linspace(0, 800, 1500)
fig, ax = plt.subplots()
ax.plot(x, f(x), label='Target Function')
#ax.set_ylim([-4, 0])  if needed to determine the limit
ax.axhline(y=0, color='k')
ax.axvline(x=0, color='k')
ax.grid()
plt.legend()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                                  Q1B
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import numpy as np
import matplotlib.pyplot as plt

def gsection(ftn, xl, xm, xr, tol=0.01):  #Tolerance modified as per question

    # Calculate the golden ratio plus one
    gr1 = 1 + (1 + np.sqrt(5)) / 2

    # Initialize function values at bounds and midpoint
    '''Adjusted to match f(x)'''
    
    fl = f(xl)
    fr = f(xr)
    fm = f(xm)

    # Initialize previous midpoint for relative error calculation
    prev_xm = None

    # Iteration counter
    iteration = 0

    # Iteratively refine xl, xr, and xm
    while (xr - xl) > tol:
        iteration += 1  # Increment iteration counter

        if (xr - xm) > (xm - xl):
            y = xm + (xr - xm) / gr1
            fy = f(y)
            if fy <= fm:  # Changed condition for minimization
                xl, fl = xm, fm
                xm, fm = y, fy
            else:
                xr, fr = y, fy
        else:
            y = xm - (xm - xl) / gr1
            fy = f(y)
            if fy <= fm:  # Changed condition for minimization
                xr, fr = xm, fm
                xm, fm = y, fy
            else:
                xl, fl = y, fy

        # Calculate relative error if previous midpoint exists
        if prev_xm is not None:
            rel_error = abs((xm - prev_xm) / xm)

            # Print current iteration, values, and relative error
            print(f'Iteration {iteration}: xl = {xl}, xm = {xm}, xr = {xr}, f(xm) = {fm}, Relative Error = {rel_error}')

        prev_xm = xm

    return xm

# Set the bounds and midpoint for the golden section search
xl, xm, xr = 0, 400, 800  # Lower bound, initial midpoint guess, upper bound

# Perform the golden section search
x_min = gsection(f(x), xl, xm, xr, tol=1e-9)
print(f'x_min: {x_min}')

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                                  Q1C
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


from sympy import symbols, exp, series

# Define the symbols
x, h = symbols('x h')

# Define the function
f = exp(-x)

# Taylor series expansion around x=1 for the given perturbation h=0.1
taylor_expansion = series(f, x, 1, 4).removeO()
taylor_expansion_with_h = taylor_expansion.subs(x, 1 + h)

print('Without the perturbation the series is', taylor_expansion) 
print('With the perturbation the series is',taylor_expansion_with_h)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                                  Q1D
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# Redefine the function and its variables for clarity
x, h = symbols('x h')
f = exp(-x)

# Initial value of x and perturbation value
x0 = 1
h_value = 0.1

# Calculate the function value at the initial x and at the perturbed x
f_x0 = f.subs(x, x0)
f_x0_perturbed = f.subs(x, x0 + h_value)

# Calculate the change in the function value
delta_f = f_x0_perturbed - f_x0

# Calculate the sensitivity
sensitivity = delta_f / h_value

# Nicely formatted print statement
print(f"Value of f(x) at x = {x0}: {f_x0.evalf():.6f}")
print(f"Value of f(x) at x = {x0 + h_value}: {f_x0_perturbed.evalf():.6f}")
print(f"Change in f(x) due to perturbation: {delta_f.evalf():.6f}")
print(f"Sensitivity of f(x) to perturbation: {sensitivity.evalf():.6f}")

'''This question can also be solved by hand'''









