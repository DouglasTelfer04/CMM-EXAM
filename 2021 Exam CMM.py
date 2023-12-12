#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 11:09:12 2023

@author: douglas
"""

##QUESTION 1 mass spring damper system
import numpy as np

# Coefficients of the polynomial
coefficients = [1, 24, 4500, 18000, 2250000]

# Solve the polynomial equation
roots = np.roots(coefficients)

# Print the roots
print("Roots:", roots)
#method based on linear algebra and eigenvalue conmputation. Coefficients are used to create a companion 
#matrix . for polynomial degree n, matrix is n by n
#Eigenvalues correspond to roots of matrix. Accuracy of result may be influenced by characteristics of 
#polynomial such as its degree and presence of multiple roots. problems can arise when coefficients have
#a large difference in magnitude. 
#The method relies on the eigenvalues of the companion matrix. If the matrix is ill-conditioned
# (close to being singular), the accuracy of the eigenvalue computation and, consequently, the roots
# may be compromised.
# In some cases, the companion matrix may become singular, making the eigenvalue computation
# challenging or resulting in undefined behavior.
#np.roots computes problems by finding eigenvalues of a constructed matrix






def newton_raphson_complex(f, df, x0, tol=1e-8, max_iter=1000):
    roots = []
    
    for _ in range(max_iter):
        x1 = x0 - f(x0) / df(x0)

        if abs(x1 - x0) < tol:
            roots.append(x1)
            x0 = x1
            break

        x0 = x1

    return roots

# Define the 4th order polynomial function
def f(x):
    return x**4 + 24*x**3 + 4500*x**2 + 18000*x + 2250000

# Define the derivative of the polynomial function
def df(x):
    return 4*x**3 + 72*x**2 + 9000*x + 18000

# Adjusted initial guesses for roots based on provided ranges
initial_guesses = [
    -2.5 + 25j, -2.5 - 25j, -10 + 62.5j, -10 - 62.5j
]

# Use Newton-Raphson for complex roots for each initial guess
all_roots = []
for guess in initial_guesses:
    roots = newton_raphson_complex(f, df, guess, max_iter=10000)
    all_roots.extend(roots)

# Display all the roots
print("Complex roots found by Newton-Raphson:")
for i, root in enumerate(all_roots, start=1):
    print(f"Root {i}: {root:.8f}")
    print(f"Function value at the root: {f(root):.8f}")



# The Newton-Raphson method for complex roots is an iterative numerical technique used to approximate
# solutions to complex-valued equations. In this implementation, the function `newton_raphson_complex` 
#takes as input a complex-valued function `f`, its derivative `df`, an initial guess `x0`, and optional
# parameters for tolerance (`tol`) and maximum iterations (`max_iter`). The method iteratively refines
# the initial guess by updating it through the formula `x_{n+1} = x_n - f(x_n) / f'(x_n)`, where `f` is
# the complex function and `f'` is its derivative.

# The algorithm continues this iterative process until either the difference between consecutive
# approximations falls below the specified tolerance (`tol`) or the maximum number of iterations 
#(`max_iter`) is reached. The updated roots are stored in the list `roots`. The initial guesses are
# provided based on the expected locations of the complex roots.

# In the provided example, the Newton-Raphson method is applied to find complex roots of a 4th order
# polynomial. The roots are computed for each initial guess, and the results, along with the
# corresponding function values at the roots, are displayed. It's important to note that the method
# may converge to different roots depending on the choice of initial guesses, and careful consideration
# of the function's behavior is required for robust application.

##Specific initial guesses were determined based on previous analysis of rough range roots were in

























##QUESTION 2
print("QUESTION 2")
m = 7850 # kg/m^3
L = 0.9 # m
E = 200*10**9 #MPa
I = 3.255*10**(-11)


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
f = lambda x: math.cosh(((2*math.pi*x)**2 * ((m*L**3)/(E*I)))**(1/4))*math.cos(((2*math.pi*x)**2 * ((m*L**3)/(E*I)))**(1/4)) + 1
approx_B = bisection(f,0.000001,0.1,25)
approx_B_2 = bisection(f,0.05,0.2,25)
print(f'approx B1: {approx_B}')
print(approx_B_2)

##Two smallest values of f determined



























##QUESTION 2 wack

#First plot graph

import matplotlib.pyplot as plt
x = np.linspace(-5,5,100)
y = np.cosh(x) * np.cos(x) + 1 
plt.plot(x,y)
plt.show()
def newton(f,Df,x0,epsilon,max_iter):
    '''
    
##Open root finding method
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

f = lambda x: np.cosh(x) * np.cos(x) + 1 
df= lambda x: (np.sinh(x)*np.cos(x) - np.sin(x) * np.cosh(x))/(np.sinh(x))**2
x0 = 4
epsilon=0.00001
max_iter=1000
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
f = lambda x: np.cosh(x) * np.cos(x) + 1
approx_phi = bisection(f,-5.5,-4,50)
print(approx_phi)






##QUESTION 3
print("QUESTION 3")
#False position method used, works well but pretty inefficient, allows intermediate
#root estimate of common point of two triangles
#First plot to visualise problem
x = np.linspace(-7,0,100)
y = x**2 + 5*x - 4
plt.plot(x,y)
plt.show()


MAX_ITER = 1000000
  
# An example function whose solution 
# is determined using Bisection Method.  
# The function is x^3 - x^2 + 2 
def func( x ): 
    return x**2 + 5*x - 4 
  
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
a =-6
b = 0
regulaFalsi(a, b) 

print("version of code with error defined")



MAX_ITER = 1000000

# An example function whose solution 
# is determined using the Regula Falsi Method.  
# The function is x^2 + 5x - 4
def func(x):
    return x**2 + 5*x - 4

# Prints root of func(x) in interval [a, b]
def regulaFalsi(a, b):
    if func(a) * func(b) >= 0:
        print("You have not assumed right a and b")
        return -1

    c = a  # Initialize result
    prev_c = 0  # Initialize previous approximation

    for i in range(MAX_ITER):
        # Find the point that touches x-axis
        c = (a * func(b) - b * func(a)) / (func(b) - func(a))

        # Check if the above-found point is a root
        if func(c) == 0:
            break

        # Calculate relative error
        rel_error = abs(c - prev_c) / abs(c) if c != 0 else 0

        # Output the current iteration, root value, and relative error
        print(f"Iteration {i + 1}: Root = {c:.8f}, Relative Error = {rel_error:.12f}")

        # Check if the relative error is below a certain threshold (e.g., 1e-8)
        if rel_error < 1e-8:
            break

        # Decide the side to repeat the steps
        elif func(c) * func(a) < 0:
            b = c
        else:
            a = c

        # Update the previous approximation
        prev_c = c

# Driver code to test above function
# Initial values assumed
a = -6
b = 0
regulaFalsi(a, b)












def inverse_quadratic_interpolation(f, x0, x1, x2, max_iter=20000000, tolerance=1e-5):
    steps_taken = 0
    while steps_taken < max_iter and abs(x1-x0) > tolerance: # last guess and new guess are v close
        fx0 = f(x0)
        fx1 = f(x1)
        fx2 = f(x2)
        L0 = (x0 * fx1 * fx2) / ((fx0 - fx1) * (fx0 - fx2))
        L1 = (x1 * fx0 * fx2) / ((fx1 - fx0) * (fx1 - fx2))
        L2 = (x2 * fx1 * fx0) / ((fx2 - fx0) * (fx2 - fx1))
        new = L0 + L1 + L2
        x0, x1, x2 = new, x0, x1
        steps_taken += 1
    return x0, steps_taken
 
f = lambda x: x**2 + 5*x - 4
 
root, steps = inverse_quadratic_interpolation(f, 4.3, 4.4, 4.5)
print ("root is:", root)
print ("steps taken:", steps)











##QUESTION 4

import numpy as np
import matplotlib.pyplot as plt
import math

# ------------------------------------------------------
# inputs

# functions that returns dy/dx
# i.e. the equation we want to solve: dy/dx = - y
lam = -10
def model(y,t):
    dydt = lam*y + (1-lam) * np.cos(t) - (1 + lam) * np.sin(t)
    return dydt

# initial conditions
t0 = 0
y0 = 1
max_error0 = 0
# total solution interval
t_final = 20
# step size
h = 0.01
# ------------------------------------------------------

# ------------------------------------------------------
# Secant method (a very compact version)
def secant_2(f, a, b, iterations):
    for i in range(iterations):
        c = a - f(a)*(b - a)/(f(b) - f(a))
        if abs(f(c)) < 1e-13:
            return c
        a = b
        b = c
    return c
# ------------------------------------------------------

# ------------------------------------------------------
# Euler implicit method

# number of steps
start = 2*np.pi
n_step = math.ceil(t_final/h)

# Definition of arrays to store the solution
y_eul = np.zeros(n_step+1)
t_eul = np.zeros(n_step+1)
max_error = np.zeros(n_step+1)

# Initialize first element of solution arrays 
# with initial condition
y_eul[0] = y0
t_eul[0] = t0 
max_error[0] = max_error0

# Populate the t array
for i in range( n_step):
    
    t_eul[i+1]  = t_eul[i]  + h

# Apply implicit Euler method n_step times
for i in range(n_step):
    F = lambda y_i_plus_1: y_eul[i] + \
            model(y_i_plus_1,t_eul[i+1])*h - y_i_plus_1
    y_eul[i+1] = secant_2(F, \
            y_eul[i],1.1*y_eul[i]+10**-3,10)
    max_error[i+1] = abs((np.sin(t_eul[i]) + np.cos(t_eul[i]) - y_eul[i]))
max_max_error = max(max_error)
print(f' max error is {max_max_error}')
        
times_of_interest = [2*np.pi, 4*np.pi]
for time_point in times_of_interest:
    index_at_time_point = int(time_point / h)
    print(f"At t = {time_point}: y = {y_eul[index_at_time_point]}")
# ------------------------------------------------------

# ------------------------------------------------------
# super refined sampling of the exact solution c*e^(-x)
# n_exact linearly spaced numbers
# only needed for plotting reference solution

# Definition of array to store the exact solution

# ------------------------------------------------------


##code to change range to 2pi to 4pi


import numpy as np
import matplotlib.pyplot as plt
import math

# Constants
LAM = -10
T_FINAL = 4 * np.pi
H = 0.2

# Function to solve
def model(y, t):
    return LAM * y + (1 - LAM) * np.cos(t) - (1 + LAM) * np.sin(t)

# Secant method
def secant_2(f, a, b, iterations):
    for i in range(iterations):
        c = a - f(a) * (b - a) / (f(b) - f(a))
        if abs(f(c)) < 1e-13:
            return c
        a, b = b, c
    return c

# Implicit Euler method
def implicit_euler_step(y_i, t_i, h):
    F = lambda y_i_plus_1: y_i + model(y_i_plus_1, t_i + h) * h - y_i_plus_1
    return secant_2(F, y_i, 1.1 * y_i + 10 ** -3, 10)

# Time array
t_eul = np.linspace(0, T_FINAL, int((T_FINAL - 2*np.pi) / H) + 1)

# Arrays initialization
y_eul, max_error = np.zeros_like(t_eul), np.zeros_like(t_eul)

# Initial condition
y_eul[0] = 1

# Implicit Euler method iterations
for i in range(len(t_eul) - 1):
    y_eul[i + 1] = implicit_euler_step(y_eul[i], t_eul[i], H)
    max_error[i + 1] = abs((np.sin(t_eul[i]) + np.cos(t_eul[i]) - y_eul[i]))

# Plotting
plt.plot(t_eul, y_eul, 'b.-', label='Implicit Euler')
plt.xlabel('x')
plt.ylabel('y(x)')
plt.legend()
plt.show()

# Print max error
max_max_error = max(max_error)
print(f'Max error is {max_max_error}')

# Print values at specific time points
times_of_interest = [2 * np.pi, 4 * np.pi]
for time_point in times_of_interest:
    index_at_time_point = int((time_point - 2 * np.pi) / H)
    print(f"At t = {time_point}: y = {y_eul[index_at_time_point]}")



# ------------------------------------------------------
# plot results
plt.plot(t_eul, y_eul , 'b.-')
plt.xlabel('x')
plt.ylabel('y(x)')
plt.show()
# ------------------------------------------------------
##Values changed to get max error in rnge 0 to 2pi and also range 2pi to 4pi
