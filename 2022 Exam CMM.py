#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 15:27:59 2023

@author: douglas
"""
##QUESTION 1: truncate approximation
import numpy as np

def truncate(N):
    sqrd = 0
    for i in range(1,N +1):
        sqrd_old = sqrd
        sqrd += 6 * 1/i**2
        
    approx = np.sqrt(sqrd)
    true_error = np.pi - approx
    Rel_error = (np.sqrt(approx - np.sqrt(sqrd_old)))/approx
    print(approx)
    print(f'true error is {true_error}')
    print(f'Relative error is {Rel_error}')
    print("")
    
truncate(10)
truncate(100)
truncate(1000)

##EXAMple solution for relative error
##Relative error defined as absolute value of (final approx - previous approx)
import numpy as np
import matplotlib.pyplot as plt
import math
N = 100
s=0
pi_n = np.zeros(N)
nn = np.zeros(N)
error_true = np.zeros(N)
error_ext = np.zeros(N)
for i in range(1,N+1):
    pi_old = (s*6.0)**0.5
    s = s + 1.0/i**2.0
    pi_n[i-1] = (s*6.0)**0.5
    nn[i-1] = i
error_true[i-1] = np.absolute(pi_n[i-1] - np.pi)
error_ext[i-1] = np.absolute(pi_n[i-1] - pi_old)
print(i,pi_n[i-1],error_true[i-1], error_ext[i-1])





#--------------------------------------------------------------------------------





##QUESTION 2: Euler method approximation

import numpy as np
import math
import matplotlib.pyplot as plt

def model(y, t):
    dydt = 10*y**2 - y**3
    return dydt

t0 = 0
y0 = 0.02
t_final = 20
h = 0.001
n_steps = math.ceil(t_final/h)
y_eul = np.zeros(n_steps + 1)
t_eul = np.zeros(n_steps + 1)
y_eul[0] = y0
t_eul[0] = t0

for i in range(n_steps):
    t_eul[i+1] = t_eul[i] + h
    slope = model(y_eul[i], t_eul[i])
    y_eul[i+1] = y_eul[i] + h * slope

# Print values at specific time points
times_of_interest = [4.9, 5, 5.1]
for time_point in times_of_interest:
    index_at_time_point = int(time_point / h)
    print(f"At t = {time_point}: y = {y_eul[index_at_time_point]}")

# Plot the results
plt.plot(t_eul, y_eul, 'b.-')
plt.xlabel('t')
plt.ylabel('y(t)')
plt.show()

## The ignition delay for initial conditions of 0.02, 0.01 and 0.005 are roughly 5s, 10s and 20s respectively

##From experimentation with step values, system starts to becomes unstable for h > 0.02
'''
##Iplicit euler method is unconditionally stable
##Implicit methods employ the use fo locations that havent been computed yet

**Stability of Numerical Methods for ODEs:**

The stability of numerical methods for solving Ordinary Differential Equations (ODEs) is a crucial consideration in ensuring the accuracy and reliability of the solutions. A stable method produces solutions that do not exhibit unbounded growth or oscillations as the numerical integration progresses. Stability is particularly important in the context of ODEs because it directly impacts the reliability of predictions and the ability to capture the behavior of the underlying physical system.

**Explicit vs. Implicit Euler Methods:**

1. **Explicit Euler Method:**
   - In the explicit Euler method, the solution at the next time step is computed solely based on the
   information from the current time step.
   - The update formula is `y_{i+1} = y_i + h * f(y_i, t_i)`, where `h` is the step size, `f` is the
   derivative function, and `y_i` and `t_i` are the solution and time at the current step.
   - Explicit methods are easy to implement but may suffer from stability issues, especially when the 
   step size is large.

2. **Implicit Euler Method:**
   - In the implicit Euler method, the solution at the next time step is obtained by solving an equation
   that involves both the current and next time steps.
   - The update formula is `y_{i+1} = y_i + h * f(y_{i+1}, t_{i+1})`, where the function `f` is evaluated
   at the next time step values.
   - Implicit methods can be more stable for certain types of ODEs, as they inherently account for
   future behavior. However, they often require solving nonlinear equations at each time step, which
   can be computationally more demanding.

**Implications for Stability and Step Size (h):**

1. **Explicit Methods:**
   - Explicit methods are conditionally stable, meaning that there is a maximum allowable step size for 
   stability.
   - The stability is often characterized by the Courant-Friedrichs-Lewy (CFL) condition, which imposes
   a restriction on the step size based on the properties of the system being modeled.
   - Large step sizes may lead to instability and numerical artifacts.

2. **Implicit Methods:**
   - Implicit methods are unconditionally stable for certain types of problems, allowing for larger
   step sizes without stability concerns.
   - However, the computational cost associated with solving implicit equations at each step can be
   higher.

**Choice of Step Size (h):**
   - The selection of the step size (h) is a trade-off between accuracy and stability.
   - Smaller step sizes generally improve accuracy but increase computational cost.
   - The stability of explicit methods is more sensitive to the choice of step size, requiring careful
   consideration to prevent instability.
   - Implicit methods often offer more flexibility in choosing step sizes, allowing for larger steps
   without sacrificing stability.

In summary, the stability of numerical methods for ODEs is a critical factor in their effectiveness.
 The explicit and implicit Euler methods illustrate different approaches, with explicit methods
 requiring careful consideration of step size for stability and implicit methods offering more 
 stability flexibility at the cost of increased computational complexity. The choice of step size
 should be made based on a balance between stability requirements and computational efficiency.

'''






##QUESTION 3 iterative equation root finding


def newton(f,Df,x0,epsilon,max_iter):
    
    
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

f = lambda x: 1/np.sin(x) + 1/4 - x
df= lambda x: -np.cos(x)/(np.sin(x))**2 - 1
x0=1
epsilon=0.00001
max_iter=100
solution = newton(f,df,x0,epsilon,max_iter)
print(f'solution: {solution}')






##closed root finding method

import math
import numpy as np
##After initially plotting function, suitable bounds were chosen for the bisection method

def bisection(f, a, b, N):
    '''Approximate solution of f(x) = 0 on interval [a,b] by bisection method.

    Parameters
    ----------
    f : function
        The function for which we are trying to approximate a solution f(x) = 0.
    a, b : numbers
        The interval in which to search for a solution. The function returns
        None if f(a) * f(b) >= 0 since a solution is not guaranteed.
    N : positive integer
        The number of iterations to implement.

    Returns
    -------
    x_N : number
        The midpoint of the Nth interval computed by the bisection method. The
        initial interval [a_0,b_0] is given by [a,b]. If f(m_n) == 0 for some
        midpoint m_n = (a_n + b_n)/2, then the function returns this solution.
        If all signs of values f(a_n), f(b_n) and f(m_n) are the same at any
        iteration, the bisection method fails and returns None.
    '''
    if f(a) * f(b) >= 0:
        print("Bisection method fails.")
        return None

    a_n = a
    b_n = b

    for n in range(1, N + 1):
        m_n = (a_n + b_n) / 2
        f_m_n = f(m_n)

        if f(a_n) * f_m_n < 0:
            a_n = a_n
            b_n = m_n
        elif f(b_n) * f_m_n < 0:
            a_n = m_n
            b_n = b_n
        elif f_m_n == 0:
            print("Found exact solution.")
            return m_n
        else:
            print("Bisection method fails.")
            return None

    return (a_n + b_n) / 2

# Example usage with your function
f = lambda x: 1/np.sin(x) + 1/4 - x
approx_phi = bisection(f, 1, 1.5, 3)
print(f"answer is {approx_phi}")
















##QUESTION 4

from sympy import Function, dsolve, Eq, Derivative, symbols

# Define symbols
t, m, b, k = symbols('t m b k')
x = Function('x')

# Define the differential equation for a mass-spring-damper system
diff_eq = Eq(m * Derivative(x(t), t, t) + b * Derivative(x(t), t) + k * x(t), 0)

# Solve the differential equation
solution = dsolve(diff_eq)

# Print the general solution
print("General Solution:")
print(solution)




from sympy import Function, dsolve, Eq, Derivative, symbols, cos, exp, sqrt
from sympy.abc import A, alpha, w

# Define symbols
t, m, b, k = symbols('t m b k')
x = Function('x')

# Given solution
A, alpha, w = symbols('A alpha w')
x_t = A * exp(-b/(2*m) * t) * cos(w * t + alpha)

# Define the relations
omega = sqrt(k/m)
damping_ratio = b/(2*m*omega)

# Substitute into the given solution
x_t = x_t.subs({w: omega, b/(2*m): damping_ratio})

# Differentiate x(t) with respect to t
dx_dt = x_t.diff(t)
dx2_dt2 = x_t.diff(t, 2)

# Substitute x(t), dx/dt, and d^2x/dt^2 into the dynamics equation
dynamics_eq = m * dx2_dt2 + b * dx_dt + k * x_t

# Simplify the result
dynamics_eq_simplified = dynamics_eq.simplify()

# Display the results
print("Given solution:")
print(x_t)

print("\nFirst derivative:")
print(dx_dt)

print("\nSecond derivative:")
print(dx2_dt2)

print("\nDynamics equation:")
print(dynamics_eq_simplified)
print('')









'''sucram'''

# SECTION A
import sympy as sym
a0=0.05
omega0 = 5
phi = 0

b = sym.Symbol('b')
m = sym.Symbol('m')
k = m*omega0**2

t = sym.Symbol('t')

omega = sym.sqrt(omega0**2-(b/(2*m))**2)

x = a0*sym.exp(-b/(2*m)*t)*sym.cos(omega*t+phi)

dxdt = sym.diff(x,t)
d2xdt2 = sym.diff(x,t,t)

func = m*d2xdt2+b*dxdt+k*x

x = sym.simplify(func)
print(f'function with substituted values is: {func}')
print(f'function simplifies to : {x}, hence it is proven that it is the general integral')
print("")









'''later parts of q'''


#equation of damped frequency is wd = sqrt(w0^2 - (b/2m)^2)

#theta(t) = A*exp(-bt/2m)
#time solved for reduction to 1% of original size of amplitude

##Only interested in amplitude, 0.01 * theta0 * exp(-b/2I * t) = theta0 * exp(-b/2I * (t + delta_t)

##by rearranging eqn: delta_t = 2m * ln(0.01)/0.1

b = 0.1
m = 1
delta_t = 2 * m *np.log(0.01)/-b
print(f'1% amplitude reached at {delta_t}')

##Solution matches up with desmos
##To halve the time simply use the same equation with b as unknown and time as 92.1034/2
time = delta_t/2
b = -2 * m *np.log(0.01)/time
print(f'damping ratio is {b}')


