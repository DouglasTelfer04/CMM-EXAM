#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 16:20:15 2023

@author: douglas
"""
##LECTURE QUESTION
import numpy as np

# Define the system of equations
def equations(x, y):
    u = x**2 + x*y - 10
    v = y + 3*x*y**2 - 57
    return u, v

# Define the Jacobian matrix
def jacobian(x, y):
    dfdx = 2*x + y
    dfdy = x
    dgdx = 3*y**2
    dgdy = 1 + 6*x*y
    return np.array([[dfdx, dfdy], [dgdx, dgdy]])

# Define the Newton-Raphson iteration
def newton_raphson(x0, y0, max_iter=100, tol=1e-6):
    for i in range(max_iter):
        u, v = equations(x0, y0)
        f = np.array([u, v])
        J = jacobian(x0, y0)
        delta = np.linalg.solve(J, -f)
        x0 += delta[0]
        y0 += delta[1]
        if np.linalg.norm(delta) < tol:
            break
    return x0, y0, i + 1

# Initial guess
x_initial, y_initial = 2, 5

# Perform Newton-Raphson iterations
x_solution, y_solution, iterations = newton_raphson(x_initial, y_initial)

# Display the results
print(f"Initial guess: x = {x_initial}, y = {y_initial}")
print(f"Newton-Raphson solution: x = {x_solution}, y = {y_solution}")
print(f"Iterations: {iterations}")

# Perform two more iterations to determine relative error
for _ in range(2):
    u, v = equations(x_solution, y_solution)
    f = np.array([u, v])
    J = jacobian(x_solution, y_solution)
    delta = np.linalg.solve(J, -f)
    x_solution += delta[0]
    y_solution += delta[1]

# Calculate relative error after two additional steps
u, v = equations(x_solution, y_solution)
relative_error = np.linalg.norm([u, v])

print(f"Relative error after two additional steps: {relative_error}")



##TUTORIAL 3 Q1
def bisection(f,a,b,N):
    # Check if a and b bound a root
    if f(a)*f(b) >= 0:
       print("a and b do not bound a root")
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
   

# we solve equation f(x)=0
f = lambda x: np.sin(x) * np.exp(x**0.1) 
# first root
approx_phi = bisection(f,2,4,10) 
print(approx_phi)
# second root
approx_phi = bisection(f,2,4,10) 
print(approx_phi)



##ERROR based
import numpy as np

def bisection(f, a, b, tol=1E-2):
    # Check if a and b bound a root
    if f(a) * f(b) >= 0:
        print("a and b do not bound a root")
        return None

    a_n = a
    b_n = b
    approx_phi_n = (a_n + b_n) / 2
    prev_approx_phi = float('inf')

    while np.abs((approx_phi_n - prev_approx_phi) / approx_phi_n) > tol:
        f_a_n = f(a_n)
        f_approx_phi_n = f(approx_phi_n)

        if f_a_n * f_approx_phi_n < 0:
            a_n = a_n
            b_n = approx_phi_n
        elif f(b_n) * f_approx_phi_n < 0:
            a_n = approx_phi_n
            b_n = b_n
        elif f_approx_phi_n == 0:
            print("Found exact solution.")
            return approx_phi_n
        else:
            print("Bisection method fails.")
            return None

        prev_approx_phi = approx_phi_n
        approx_phi_n = (a_n + b_n) / 2

    return approx_phi_n

# Solve equation f(x) = 0
f = lambda x: x**2 + 4*x - 12

# First root
approx_phi_1 = bisection(f, -7,-2)
print(f"Approximate root: {approx_phi_1}")

# Second root
approx_phi_2 = bisection(f, 0, 3)
print(f"Approximate root: {approx_phi_2}")






##NEWTON Method
def newton(f,Df,x0,epsilon,max_iter):
    xn = x0
    for n in range(0,max_iter):
        fxn = f(xn)
        if abs(fxn) < epsilon:
            print('Found solution after',n,'iterations.:', xn)
            return xn
        
        Dfxn = Df(xn)
        if Dfxn == 0:
            print('Zero derivative. No solution found.')
            return None
        xn = xn - fxn/Dfxn
    print('Exceeded maximum iterations. No solution found.')
    return None
f =lambda x: x**2 + 4*x -12
df=lambda x: 2*x + 4
x0=1
epsilon=0.00001
max_iter=100
solution = newton(f,df,x0,epsilon,max_iter)




##SCIPY optimize

from scipy import optimize
def f(x):
    return x**2 + 4*x - 12
solution = optimize.newton(f, -10)
print(solution)

solution = optimize.newton(f, 1.5)
print(solution)












##TUTORIAL 4
# import modules ----------------------------------------
import math
import numpy as np
import matplotlib.pyplot as plt


# definition of function --------------------------------

# Newton 
def newton2(f,Df,x0,N):
    xn = x0
    for n in range(0,N):
        fxn = f(xn)
        Dfxn = Df(xn)
        if Dfxn == 0:
            print('Zero derivative. No solution found.')
            return None
        xn = xn - fxn/Dfxn
    return xn


# Secant
def secant(f,a,b,N):

    if f(a)*f(b) >= 0:
        print("Secant method fails.")
        return None
    a_n = a
    b_n = b
    for n in range(1,N+1):
        m_n = a_n - f(a_n)*(b_n - a_n)/(f(b_n) - f(a_n))
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
            print("Secant method fails.")
            return None
    return a_n - f(a_n)*(b_n - a_n)/(f(b_n) - f(a_n))


# ----------------------------------------------------------------------------------------
# main program ------------------------------------
n_max=20

n_array_N   = np.zeros(n_max-1)
sol_array_N = np.zeros(n_max-1)
fun_array_N = np.zeros(n_max-1)

n_array_S   = np.zeros(n_max-1)
sol_array_S = np.zeros(n_max-1)
fun_array_S = np.zeros(n_max-1)

# ----------------------------------------------------------------------------------------

f = lambda x: x**2 + 4*x - 12
df= lambda x: 2*x + 4

# Initial guess for Newton
x0=1

# Initial a and b for secant
a=1
b=3

for i in range(1,n_max):
    solution = newton2(f,df,x0,i)
    n_array_N[i-1] = i  
    sol_array_N[i-1] = solution
    fun_array_N[i-1] = np.absolute(f(solution))

    solution = secant(f,a,b,i)
    n_array_S[i-1] = i  
    sol_array_S[i-1] = solution
    fun_array_S[i-1] = np.absolute(f(solution))

plt.figure()
plt.plot(n_array_N,sol_array_N, '-o',n_array_S,sol_array_S, '-o')
plt.xlabel("Number of iterations")
plt.ylabel("Solution")
plt.xlim(0,n_max)

plt.figure()
# plot the error defined as f(solution)
plt.semilogy(n_array_N,fun_array_N, '-o',n_array_S,fun_array_S, '-o')
# plot a couple of scaling lines to assess convergence rate
plt.semilogy(n_array_S,np.exp(-2.0*n_array_S))
plt.semilogy(n_array_S,np.exp(-2.5*n_array_S))
plt.xlabel("Number of iterations")
plt.ylabel("Error, defined as f(solution)")

plt.xlim(0,n_max)


# ----------------------------------------------------------------------------------------
f = lambda x: math.sin(x) * math.exp(x**0.1)
df= lambda x: (math.exp(x**0.1)*math.sin(x))/(10*x**(9/10))+math.exp(x**0.1)*math.cos(x)

# Initial guess for Newton
x0=4

# Initial a and b for secant
a=1
b=4

for i in range(1,n_max):
    solution = newton2(f,df,x0,i)
    n_array_N[i-1] = i  
    sol_array_N[i-1] = solution
    fun_array_N[i-1] = np.absolute(f(solution))

    solution = secant(f,a,b,i)
    n_array_S[i-1] = i  
    sol_array_S[i-1] = solution
    fun_array_S[i-1] = np.absolute(f(solution))

plt.figure()
plt.plot(n_array_N,sol_array_N, '-o',n_array_S,sol_array_S, '-o')
plt.xlabel("Number of iterations")
plt.ylabel("Solution")
plt.xlim(0,n_max)

plt.figure()
# plot the error definrd as f(solution)
plt.semilogy(n_array_N,fun_array_N, '-o',n_array_S,fun_array_S, '-o')
# plot a couple of scaling lines to assess convergence rate
plt.semilogy(n_array_S,np.exp(-2.0*n_array_S))
plt.semilogy(n_array_S,np.exp(-2.5*n_array_S))
plt.xlabel("Number of iterations")
plt.ylabel("Error, defined as f(solution)")
plt.xlim(0,n_max)




plt.show()





##EXERCISE 2 polynomial deflation
import numpy as np

def poly_iter(A, t):
    # compute q(x) = p(x)/(x-t) and residual r
    # array A contains coefficients of p(x) 
    n = len(A)-1
    # q: array of integers to store coefficients of q(x)
    q=np.zeros(n,dtype=np.int8)
    r = A[n]
    for a in reversed(range(n)):
        s=A[a]
        q[a]=r
        r = s + r * t
    print('----------------------------------------')
    print('Coefficients a0, a1, a2, ..., an')
    print('of quotient a0+a1*x+a2*x^2+...an*x^n:') 
    print(q)
    print('----------------------------------------')
    print('Residual:')
    print(r)
    print('----------------------------------------')
    return []

#A = np.array([ -24, 2, 1])
#t = 4

A = np.array([ -42, 0, -12 ,1])
t=3

poly_iter(A,t)





##WEEK 5 Gauss Elimenation
import numpy as np

def linearsolver(A,b):
    n = len(A)

    #Initialise solution vector as an empty array
    x = np.zeros(n)

    #Join A and use concatenate to form an augmented coefficient matrix
    M = np.concatenate((A,b.T), axis=1)

    for k in range(n):
        for i in range(k,n):
            if abs(M[i][k]) > abs(M[k][k]):
                M[[k,i]] = M[[i,k]]
            else:
                pass
                for j in range(k+1,n):
                    q = M[j][k] / M[k][k]
                    for m in range(n+1):
                        M[j][m] +=  -q * M[k][m]

    #Python starts indexing with 0, so the last element is n-1
    x[n-1] =M[n-1][n]/M[n-1][n-1]

    #We need to start at n-2, because of Python indexing
    for i in range (n-2,-1,-1):
        z = M[i][n]
        for j in range(i+1,n):
            z = z  - M[i][j]*x[j]
        x[i] = z/M[i][i]

    return x

#Initialise the matrices to be solved.
#A=np.array([[10., 15., 25],[4., 5., 6], [25, 3, 8]])
#b=np.array([[34., 25., 15]])
#print(linearsolver(A,b))

A=np.array([[70., 1., 0],[60., -1., 1.], [40, 0, -1]])
b=np.array([[636.7, 518.6, 307.4]])
print(linearsolver(A,b))








# importing modules
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import interpolate

# importing data from file data_reg.py
import data_reg
# creating simpler names for variables
x = data_reg.x
y = data_reg.y

# array containing the points where we want
# to evaluate the fit
x_fit = np.linspace(0,1,num=64)


# ----------------------------------------------------
# Linear regression using least squares 
# with formulas presented in Lecture

n = len(x)
a_1 = (n*np.sum(x*y) - np.sum(x)*np.sum(y))/(n * np.sum(x**2) - (np.sum(x))**2)
a_0 = np.mean(y) - a_1*np.mean(x) 

# evaluate the linear regression at the desired points 
y_reg_lin = a_0 + a_1 * x_fit 

# Print coefficients of linear regression:
print('Coefs a_1 and a_0 (our implementation): ', a_1,a_0)
# ----------------------------------------------------

# ----------------------------------------------------
# Linear regression using python functions

# compute the coefficients for the linear regression
coef = np.polyfit(x,y,1)
# generate the linear function that fits the data 
f_reg_lin = np.poly1d(coef)

# evaluate the linear regression at the desired points 
y_reg_lin_py = f_reg_lin(x_fit)

# Print coefficients of linear regression:
print('Coefs a_1 and a_0 (python function): ', coef)
# ----------------------------------------------------


# plot results
plt.figure()
plt.plot(x,y,'gh',ms=5)
plt.plot(x_fit,y_reg_lin,'b-')
plt.xlabel('x')
plt.ylabel('y')

plt.figure()
plt.plot(x,y,'gh',ms=5)
plt.plot(x_fit,y_reg_lin_py,'r-')
plt.xlabel('x')
plt.ylabel('y')


plt.show()




##WEEK 5 Linear algebra: GAUSS


import numpy as np

def linearsolver(A,b):
    n = len(A)

    #Initialise solution vector as an empty array
    x = np.zeros(n)

    #Join A and use concatenate to form an augmented coefficient matrix
    M = np.concatenate((A,b.T), axis=1)

    for k in range(n):
        for i in range(k,n):
            if abs(M[i][k]) > abs(M[k][k]):
                M[[k,i]] = M[[i,k]]
            else:
                pass
                for j in range(k+1,n):
                    q = M[j][k] / M[k][k]
                    for m in range(n+1):
                        M[j][m] +=  -q * M[k][m]

    #Python starts indexing with 0, so the last element is n-1
    x[n-1] =M[n-1][n]/M[n-1][n-1]

    #We need to start at n-2, because of Python indexing
    for i in range (n-2,-1,-1):
        z = M[i][n]
        for j in range(i+1,n):
            z = z  - M[i][j]*x[j]
        x[i] = z/M[i][i]

    return x

#Initialise the matrices to be solved.
#A=np.array([[10., 15., 25],[4., 5., 6], [25, 3, 8]])
#b=np.array([[34., 25., 15]])
#print(linearsolver(A,b))

A=np.array([[70, 1, 0],[60, -1, 1], [40, 0, -1]])
b=np.array([[636.7, 518.6, 307.4]])
print(linearsolver(A,b))









import numpy as np

def linearsolver(A,b):
    n = len(A)

    #Initialise solution vector as an empty array
    x = np.zeros(n)

    #Join A and use concatenate to form an augmented coefficient matrix
    M = np.concatenate((A,b.T), axis=1)

    for k in range(n):
        for i in range(k,n):
            if abs(M[i][k]) > abs(M[k][k]):
                M[[k,i]] = M[[i,k]]
            else:
                pass
                for j in range(k+1,n):
                    q = M[j][k] / M[k][k]
                    for m in range(n+1):
                        M[j][m] +=  -q * M[k][m]

    #Python starts indexing with 0, so the last element is n-1
    x[n-1] =M[n-1][n]/M[n-1][n-1]

    #We need to start at n-2, because of Python indexing
    for i in range (n-2,-1,-1):
        z = M[i][n]
        for j in range(i+1,n):
            z = z  - M[i][j]*x[j]
        x[i] = z/M[i][i]

    return x

#Initialise the matrices to be solved.
#A=np.array([[10., 15., 25],[4., 5., 6], [25, 3, 8]])
#b=np.array([[34., 25., 15]])
#print(linearsolver(A,b))

A=np.array([[30, -20, 0],[-20, 30, -10], [0, -10, 10]])
b=np.array([[19.62, 24.525, 29.43]])
print(linearsolver(A,b))




import numpy as np

# Given values
k1 = 10
k2 = 20
k3 = 20
m1 = 2
m2 = 2.5
m3 = 3
g = 9.8  # Acceleration due to gravity

# Constructing the coefficient matrix
A = np.array([[k1 + k2, -k2, 0],
              [-k2, k2 + k3, -k3],
              [0, -k3, k3]])

# Constructing the right-hand side vector
B = np.array([m1 * g, m2 * g, m3 * g])

# Solving for the deflections
deflections = np.linalg.solve(A, B)

# Display the results
for i, deflection in enumerate(deflections, start=1):
    print(f'Deflection x{i} = {deflection:.4f} meters')




#----------------------------------------------
##ODES

# importing modules
import numpy as np
import matplotlib.pyplot as plt
import math

# ------------------------------------------------------
# inputs

# functions that returns dy/dx
# i.e. the equation we want to solve: dy/dx = - y
def model(y,x):
    k= - 1
    dydx = k * y
    return dydx

# initial conditions
x0 = 0
y0 = 1
# total solution interval
x_final = 1
# step size
h = 0.1
# ------------------------------------------------------

# ------------------------------------------------------
# Euler method

# number of steps
n_step = math.ceil(x_final/h)

# Definition of arrays to store the solution
y_eul = np.zeros(n_step+1)
x_eul = np.zeros(n_step+1)

# Initialize first element of solution arrays 
# with initial condition
y_eul[0] = y0
x_eul[0] = x0 

# Populate the x array
for i in range(n_step):
    x_eul[i+1]  = x_eul[i]  + h

# Apply Euler method n_step times
for i in range(n_step):
    # compute the slope using the differential equation
    slope = model(y_eul[i],x_eul[i]) 
    # use the Euler method
    y_eul[i+1] = y_eul[i] + h * slope  
# ------------------------------------------------------

# ------------------------------------------------------
# super refined sampling of the exact solution 
# n_exact linearly spaced numbers
# only needed for plotting reference solution

# Definition of array to store the exact solution
n_exact = 1000
x_exact = np.linspace(0,x_final,n_exact+1) 
y_exact = np.zeros(n_exact+1)

# exact values of the solution
for i in range(n_exact+1):
    y_exact[i] = y0 * math.exp(-x_exact[i])
# ------------------------------------------------------

# ------------------------------------------------------
# print results on screen
print ('Solution: step x y-eul y-exact error%')
for i in range(n_step+1):
    print(i,x_eul[i],y_eul[i], y0 * math.exp(-x_eul[i]),
            (y_eul[i]- y0 * math.exp(-x_eul[i]))/ 
            (y0 * math.exp(-x_eul[i])) * 100)
# ------------------------------------------------------

# ----------------------------------------------------

# ------------------------------------------------------
# plot results
plt.plot(x_eul, y_eul , 'b.-',x_exact, y_exact , 'r-')
plt.xlabel('x')
plt.ylabel('y(x)')
plt.show()
# ------------------------------------------------------



    
    
    
    
    
    
# importing modules
import numpy as np
import matplotlib.pyplot as plt
import math

# ------------------------------------------------------
# inputs

# functions that returns dy/dx
# i.e. the equation we want to solve: dy/dx = - y
def model(y,x):
    k= - 1
    dydx = k * y
    return dydx

# initial conditions
x0 = 0
y0 = 1
# total solution interval
x_final = 1
# step size
h = 0.2
# ------------------------------------------------------

# ------------------------------------------------------
# Fourth Order Runge-Kutta method

# number of steps
n_step = math.ceil(x_final/h)

# Definition of arrays to store the solution
y_rk = np.zeros(n_step+1)
x_rk = np.zeros(n_step+1)

# Initialize first element of solution arrays 
# with initial condition
y_rk[0] = y0
x_rk[0] = x0 

# Populate the x array
for i in range(n_step):
    x_rk[i+1]  = x_rk[i]  + h

# Apply RK method n_step times
for i in range(n_step):
   
    # Compute the four slopes
    x_dummy = x_rk[i]
    y_dummy = y_rk[i]
    k1 =  model(y_dummy,x_dummy)
    
    x_dummy = x_rk[i]+h/2
    y_dummy = y_rk[i] + k1 * h/2
    k2 =  model(y_dummy,x_dummy)

    x_dummy = x_rk[i]+h/2
    y_dummy = y_rk[i] + k2 * h/2
    k3 =  model(y_dummy,x_dummy)

    x_dummy = x_rk[i]+h
    y_dummy = y_rk[i] + k3 * h
    k4 =  model(y_dummy,x_dummy)

    # compute the slope as weighted average of four slopes
    slope = 1/6 * k1 + 2/6 * k2 + 2/6 * k3 + 1/6 * k4 

    # use the RK method
    y_rk[i+1] = y_rk[i] + h * slope  
# ------------------------------------------------------

# ------------------------------------------------------
# super refined sampling of the exact solution c*e^(-x)
# n_exact linearly spaced numbers
# only needed for plotting reference solution

# Definition of array to store the exact solution
n_exact = 1000
x_exact = np.linspace(0,x_final,n_exact+1) 
y_exact = np.zeros(n_exact+1)

# exact values of the solution
for i in range(n_exact+1):
    y_exact[i] = y0 * math.exp(-x_exact[i])
# ------------------------------------------------------

# ------------------------------------------------------
# print results on screen
print ('Solution: step x y-eul y-exact error%')
for i in range(n_step+1):
    print(i,x_rk[i],y_rk[i], y0 * math.exp(-x_rk[i]),
            (y_rk[i]- y0 * math.exp(-x_rk[i]))/ 
            (y0 * math.exp(-x_rk[i])) * 100)
# ------------------------------------------------------


# ------------------------------------------------------
# plot results
plt.plot(x_rk, y_rk , 'b.-',x_exact, y_exact , 'r-')
plt.xlabel('x')
plt.ylabel('y(x)')
plt.show()
# ------------------------------------------------------















##Tutorial 7

##EULER method and verified
# importing modules
import numpy as np
import matplotlib.pyplot as plt
import math

# ------------------------------------------------------
# inputs

# functions that returns dy/dx
# i.e. the equation we want to solve: dy/dx = - y
def model(y,x):
    
    dydx = y*(1-y)
    return dydx

# initial conditions
x0 = 0
y0 = math.exp(-4)/(math.exp(-4) + 1)
# total solution interval
x_final = 10
# step size
h = 0.5
# ------------------------------------------------------

# ------------------------------------------------------
# Euler method

# number of steps
n_step = math.ceil(x_final/h)

# Definition of arrays to store the solution
y_eul = np.zeros(n_step+1)
x_eul = np.zeros(n_step+1)

# Initialize first element of solution arrays 
# with initial condition
y_eul[0] = y0
x_eul[0] = x0 

# Populate the x array
for i in range(n_step):
    x_eul[i+1]  = x_eul[i]  + h

# Apply Euler method n_step times
for i in range(n_step):
    # compute the slope using the differential equation
    slope = model(y_eul[i],x_eul[i]) 
    # use the Euler method
    y_eul[i+1] = y_eul[i] + h * slope  
# ------------------------------------------------------

# ------------------------------------------------------
# super refined sampling of the exact solution 
# n_exact linearly spaced numbers
# only needed for plotting reference solution

# Definition of array to store the exact solution
n_exact = 1000
x_exact = np.linspace(0,x_final,n_exact+1) 
y_exact = np.zeros(n_exact+1)

# exact values of the solution
for i in range(n_exact+1):
    y_exact[i] = math.exp(x_exact[i] -4)/(math.exp(x_exact[i] -4) + 1)
# ------------------------------------------------------




# ------------------------------------------------------
# plot results
plt.plot(x_eul, y_eul , 'b.-',x_exact, y_exact , 'r-')
plt.xlabel('x')
plt.ylabel('y(x)')
plt.show()
# ------------------------------------------------------








##RUNGE KUTTA method and verified

# importing modules
import numpy as np
import matplotlib.pyplot as plt
import math

# ------------------------------------------------------
# inputs

# functions that returns dy/dx
# i.e. the equation we want to solve: dy/dx = - y
def model(y,x):
    
    dydx = y* (1 - y)
    return dydx

# initial conditions
x0 = 0
y0 = math.exp(-4)/(math.exp(-4) + 1)
# total solution interval
x_final = 10
# step size
h = 0.5
# ------------------------------------------------------

# ------------------------------------------------------
# Fourth Order Runge-Kutta method

# number of steps
n_step = math.ceil(x_final/h)

# Definition of arrays to store the solution
y_rk = np.zeros(n_step+1)
x_rk = np.zeros(n_step+1)

# Initialize first element of solution arrays 
# with initial condition
y_rk[0] = y0
x_rk[0] = x0 

# Populate the x array
for i in range(n_step):
    x_rk[i+1]  = x_rk[i]  + h

# Apply RK method n_step times
for i in range(n_step):
   
    # Compute the four slopes
    x_dummy = x_rk[i]
    y_dummy = y_rk[i]
    k1 =  model(y_dummy,x_dummy)
    
    x_dummy = x_rk[i]+h/2
    y_dummy = y_rk[i] + k1 * h/2
    k2 =  model(y_dummy,x_dummy)

    x_dummy = x_rk[i]+h/2
    y_dummy = y_rk[i] + k2 * h/2
    k3 =  model(y_dummy,x_dummy)

    x_dummy = x_rk[i]+h
    y_dummy = y_rk[i] + k3 * h
    k4 =  model(y_dummy,x_dummy)

    # compute the slope as weighted average of four slopes
    slope = 1/6 * k1 + 2/6 * k2 + 2/6 * k3 + 1/6 * k4 

    # use the RK method
    y_rk[i+1] = y_rk[i] + h * slope  
# ------------------------------------------------------

# ------------------------------------------------------
# super refined sampling of the exact solution c*e^(-x)
# n_exact linearly spaced numbers
# only needed for plotting reference solution

# Definition of array to store the exact solution
n_exact = 1000
x_exact = np.linspace(0,x_final,n_exact+1) 
y_exact = np.zeros(n_exact+1)

# exact values of the solution
for i in range(n_exact+1):
    y_exact[i] = math.exp(x_exact[i] -4)/(math.exp(x_exact[i] -4) + 1)
# ------------------------------------------------------

# ------------------------------------------------------
'''# print results on screen
print ('Solution: step x y-eul y-exact error%')
for i in range(n_step+1):
    print(i,x_rk[i],y_rk[i], y0 * math.exp(-x_rk[i]),
            (y_rk[i]- y0 * math.exp(-x_rk[i]))/ 
            (y0 * math.exp(-x_rk[i])) * 100)
# ------------------------------------------------------
'''


# ------------------------------------------------------
# plot results
plt.plot(x_rk, y_rk , 'b.-',x_exact, y_exact , 'r-')
plt.xlabel('x')
plt.ylabel('y(x)')
plt.show()
# ------------------------------------------------------










# importing modules
import numpy as np
import matplotlib.pyplot as plt
import math

# ------------------------------------------------------
# inputs

# functions that returns dy/dx
# i.e. the equation we want to solve: dy/dx = - y
def model(y, x):
    dydx = y * (1 - y)
    return dydx

# initial conditions
x0 = 0
y0 = math.exp(-4) / (math.exp(-4) + 1)
# total solution interval
x_final = 10
# step size
h = 0.2
# ------------------------------------------------------

# ------------------------------------------------------
# Fourth Order Runge-Kutta method

# number of steps
n_step = math.ceil(x_final / h)

# Definition of arrays to store the solution
y_rk = np.zeros(n_step + 1)
x_rk = np.zeros(n_step + 1)

# Initialize first element of solution arrays 
# with initial condition
y_rk[0] = y0
x_rk[0] = x0

# Populate the x array
for i in range(n_step):
    x_rk[i + 1] = x_rk[i] + h

# Apply RK method n_step times
for i in range(n_step):
   
    # Compute the four slopes
    x_dummy = x_rk[i]
    y_dummy = y_rk[i]
    k1 = model(y_dummy, x_dummy)
    
    x_dummy = x_rk[i] + h / 2
    y_dummy = y_rk[i] + k1 * h / 2
    k2 = model(y_dummy, x_dummy)

    x_dummy = x_rk[i] + h / 2
    y_dummy = y_rk[i] + k2 * h / 2
    k3 = model(y_dummy, x_dummy)

    x_dummy = x_rk[i] + h
    y_dummy = y_rk[i] + k3 * h
    k4 = model(y_dummy, x_dummy)

    # compute the slope as weighted average of four slopes
    slope = 1 / 6 * k1 + 2 / 6 * k2 + 2 / 6 * k3 + 1 / 6 * k4 

    # use the RK method
    y_rk[i + 1] = y_rk[i] + h * slope  
# ------------------------------------------------------

# ------------------------------------------------------
# super refined sampling of the exact solution c*e^(-x)
# n_exact linearly spaced numbers
# only needed for plotting reference solution

# Extend the range for exact solution to cover x_final
n_exact = 50
x_exact = np.linspace(0, x_final, n_exact + 1)
y_exact = np.zeros(n_exact + 1)

# exact values of the solution
for i in range(n_exact + 1):
    y_exact[i] = math.exp(x_exact[i] - 4) / (math.exp(x_exact[i] - 4) + 1)
# ------------------------------------------------------

# ------------------------------------------------------
# Calculate RMSE
rmse = np.sqrt(np.mean((y_rk - y_exact[:n_step + 1])**2))

# print RMSE on screen
print(f'Root Mean Squared Error (RMSE): {rmse}')
# ------------------------------------------------------

# ------------------------------------------------------
# plot results
plt.plot(x_rk, y_rk, 'b.-', label='Numerical Solution')
plt.plot(x_exact, y_exact, 'r-', label='Exact Solution')
plt.xlabel('x')
plt.ylabel('y(x)')
plt.legend()
plt.show()
# ------------------------------------------------------
















# importing modules
import numpy as np
import matplotlib.pyplot as plt
import math

# ------------------------------------------------------
# inputs

# functions that returns dy/dx
# i.e. the equation we want to solve: dy/dx = - y
def model(y, x):
    dydx = y * (1 - y)
    return dydx

# initial conditions
x0 = 0
y0 = math.exp(-4) / (math.exp(-4) + 1)
# total solution interval
x_final = 10
# step size
h = 0.2
# ------------------------------------------------------

# ------------------------------------------------------
# Euler method

# number of steps
n_step = math.ceil(x_final / h)

# Definition of arrays to store the solution
y_eul = np.zeros(n_step + 1)
x_eul = np.zeros(n_step + 1)

# Initialize first element of solution arrays 
# with initial condition
y_eul[0] = y0
x_eul[0] = x0

# Populate the x array
for i in range(n_step):
    x_eul[i + 1] = x_eul[i] + h

# Apply Euler method n_step times
for i in range(n_step):
    # compute the slope using the differential equation
    slope = model(y_eul[i], x_eul[i]) 
    # use the Euler method
    y_eul[i + 1] = y_eul[i] + h * slope  
# ------------------------------------------------------

# ------------------------------------------------------
# super refined sampling of the exact solution 
# n_exact linearly spaced numbers
# only needed for plotting reference solution

# Extend the range for exact solution to cover x_final
n_exact = 50
x_exact = np.linspace(0, x_final, n_exact + 1)
y_exact = np.zeros(n_exact + 1)

# exact values of the solution
for i in range(n_exact + 1):
    y_exact[i] = math.exp(x_exact[i] - 4) / (math.exp(x_exact[i] - 4) + 1)
# ------------------------------------------------------

# ------------------------------------------------------
# Calculate RMSE
rmse = np.sqrt(np.mean((y_eul - y_exact[:n_step + 1])**2))

# print RMSE on screen
print(f'Root Mean Squared Error (RMSE): {rmse}')
# ------------------------------------------------------

# ------------------------------------------------------
# plot results
plt.plot(x_eul, y_eul, 'b.-', label='Numerical Solution')
plt.plot(x_exact, y_exact, 'r-', label='Exact Solution')
plt.xlabel('x')
plt.ylabel('y(x)')
plt.legend()
plt.show()
# ------------------------------------------------------










'''The order of a numerical method indicates how the error of the method scales with the step size.
 A method is considered k-th order accurate if the global error is proportional to the k-th power of 
 the step size (h).

In the case of the Euler method and the fourth-order Runge-Kutta (RK4) method:

1. Euler Method (First Order):
   - Local truncation error: proportional to h^2.
   - Global error: proportional to h.
   - Halving the step size (h) roughly halves the global error.

2. RK4 Method (Fourth Order):
   - Local truncation error: proportional to h^5.
   - Global error: proportional to h^4.
   - Halving the step size (h) reduces the global error by a factor of 16 (2^4).

Therefore, when comparing the RMSE for the Euler method and RK4 method for various step sizes, you
 should observe that reducing the step size for RK4 leads to a more rapid decrease in error compared
 to the Euler method. This aligns with the expected orders of accuracy for each method.
'''








##Turorial 7 solution

# importing modules
import numpy as np
import matplotlib.pyplot as plt
import math


# ------------------------------------------------------
# functions that returns dy/dx
# i.e. the equation we want to solve
def model(y,x):
    dydx = y*(1.0-y)
    return dydx
# ------------------------------------------------------

# ------------------------------------------------------
# inputs
# initial conditions
x0 = 0
y0 = math.exp(-4)/(math.exp(-4) + 1)
# total solution interval
x_final = 10
# ------------------------------------------------------

h_values = np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1, 2, 5])
RMQE_eul = np.zeros(len(h_values))
RMQE_rk  = np.zeros(len(h_values))

for hh in range(0,len(h_values)):

         # ------------------------------------------------------
         # step size
         h = h_values[hh]
         # number of steps
         n_step = math.ceil(x_final/h)
         # ------------------------------------------------------
         
         # ------------------------------------------------------
         # Euler method
         
         # Definition of arrays to store the solution
         x_eul = np.zeros(n_step+1)
         y_eul = np.zeros(n_step+1)
         e_eul = np.zeros(n_step+1)
         
         # Initialize first element of solution arrays 
         # with initial condition
         y_eul[0] = y0
         x_eul[0] = x0 
         
         # Populate the x array
         for i in range(n_step):
             x_eul[i+1]  = x_eul[i]  + h
         
         # Apply Euler method n_step times
         for i in range(n_step):
             # compute the slope using the differential equation
             slope = model(y_eul[i],x_eul[i]) 
             # use the Euler method
             y_eul[i+1] = y_eul[i] + h * slope 
         # ------------------------------------------------------
         
         
         # ------------------------------------------------------
         # Fourth Order Runge-Kutta method
         
         # Definition of arrays to store the solution
         x_rk = np.zeros(n_step+1)
         y_rk = np.zeros(n_step+1)
         e_rk = np.zeros(n_step+1)
         
         # Initialize first element of solution arrays 
         # with initial condition
         y_rk[0] = y0
         x_rk[0] = x0 
         
         # Populate the x array
         for i in range(n_step):
             x_rk[i+1]  = x_rk[i]  + h
         
         # Apply RK method n_step times
         for i in range(n_step):
            
             # Compute the four slopes
             x_dummy = x_rk[i]
             y_dummy = y_rk[i]
             k1 =  model(y_dummy,x_dummy)
             
             x_dummy = x_rk[i]+h/2
             y_dummy = y_rk[i] + k1 * h/2
             k2 =  model(y_dummy,x_dummy)
         
             x_dummy = x_rk[i]+h/2
             y_dummy = y_rk[i] + k2 * h/2
             k3 =  model(y_dummy,x_dummy)
         
             x_dummy = x_rk[i]+h
             y_dummy = y_rk[i] + k3 * h
             k4 =  model(y_dummy,x_dummy)
         
             # compute the slope as weighted average of four slopes
             slope = 1/6 * k1 + 2/6 * k2 + 2/6 * k3 + 1/6 * k4 
         
             # use the RK method
             y_rk[i+1] = y_rk[i] + h * slope  
         # ------------------------------------------------------
         
         
         # ------------------------------------------------------
         # Compute error
         e_eul = y_eul - np.exp(x_eul-4)/(np.exp(x_eul-4) + 1)
         e_rk  = y_rk  - np.exp(x_rk -4)/(np.exp(x_rk -4) + 1)
         
         # Root mean square error
         RMQE_eul[hh] = (np.mean(e_eul**2.0))**0.5
         RMQE_rk[hh]  = (np.mean(e_rk **2.0))**0.5
         # ------------------------------------------------------
         
         # ------------------------------------------------------
         # very refined sampling of the exact solution c*e^(-x)
         # n_exact linearly spaced numbers
         # only needed for plotting exact solution
         
         # Definition of array to store the exact solution
         n_exact = 1000
         x_exact = np.linspace(0,x_final,n_exact+1) 
         y_exact = np.zeros(n_exact+1)
         
         # exact values of the solution
         for i in range(n_exact+1):
             y_exact[i] = math.exp(x_exact[i]-4)/(math.exp(x_exact[i]-4) + 1)
         # ------------------------------------------------------
         
         # ------------------------------------------------------
         # plot results in the loop
         plt.figure()
         plt.plot(x_eul, y_eul , 'b.-', label='Eul')
         plt.plot(x_rk, y_rk , 'g*-', label='RK4')
         plt.plot(x_exact, y_exact , 'r-', label='Exact')
         plt.xlabel('x')
         plt.ylabel('y(x)')
         st1 = 'Solution for h = '+str(h) 
         plt.legend(title=st1)
         st2 = 'Solution_h_'+str(h)+'.pdf'
         plt.savefig(st2)
         
         plt.figure()
         plt.plot(x_eul, e_eul , 'b.-', label='Eul')
         plt.plot(x_rk, e_rk , 'g*-', label='RK4')
         plt.xlabel('x')
         plt.ylabel('error(x)')
         st1 = 'Error for h = '+str(h) 
         plt.legend(title=st1)
         st2 = 'Error_h_'+str(h)+'.pdf'
         plt.savefig(st2)
         # ------------------------------------------------------


# ------------------------------------------------------
# plot outside the loop
plt.figure()
plt.loglog(h_values, RMQE_eul , 'b.', label='Eul num sol')
plt.loglog(h_values, RMQE_rk  , 'g*', label='RK4 num sol')
plt.loglog(h_values, 0.1*h_values , 'b-', label='h^1')
plt.loglog(h_values, 1e-3*h_values**4.0 , 'g--', label='h^4')
plt.xlabel('h')
plt.ylabel('RMQE')
st1 = 'Error vs h'
plt.legend(title=st1)
st2 = 'Error_vs_h.pdf'
plt.savefig(st2)

# ------------------------------------------------------

         
#plt.show()






















##Tutorial 8

# importing modules
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.integrate import solve_ivp

sig = 10
B = 8/3
p = 28
# ------------------------------------------------------
# functions that returns dy/dx
# i.e. the equation we want to solve: 
# dy_j/dx = f_j(x,y_j) (j=[1,2] in this case)
def model(x,y):
    y_1 = y[0]
    y_2 = y[1]
    y_3 = y[2]
    f_1 = sig * (y_2 - y_1)
    f_2 = p*y_1 - y_2 - y_1 * y_3
    f_3 = -B * y_3 + y_1 * y_2
    return [f_1 , f_2, f_3]
# ------------------------------------------------------

# ------------------------------------------------------
# initial conditions
x0 = 0
y0_1 = 5
y0_2 = 5
y0_3 = 5
# total solution interval
t_final = 30
# step size
# not needed here. The solver solve_ivp 
# will take care of finding the appropriate step 
# ------------------------------------------------------

# ------------------------------------------------------
# Apply solve_ivp method
y = solve_ivp(model, [0 , t_final] ,[y0_1 , y0_2, y0_3])
# ------------------------------------------------------

# ------------------------------------------------------
# plot results
width = 6
height = 2
t = y.t
fig1, ax1 = plt.subplots(figsize = (width, height))
ax1.plot(t, y.y[0]) 
ax1.set_title('y1')

fig1, ax1 = plt.subplots(figsize = (width, height))
ax1.plot(t, y.y[1]) 
ax1.set_title('y2')



fig1, ax1 = plt.subplots(figsize = (width, height))
ax1.plot(t, y.y[2]) 
ax1.set_title('y3')


fig1, ax1 = plt.subplots(figsize = (width, height))
ax1.plot(y.y[0], y.y[1]) 
ax1.set_title('absciss y1,y2')

fig1, ax1 = plt.subplots(figsize = (width, height))
ax1.plot(y.y[0], y.y[2]) 
ax1.set_title('abscissa y1,y3')









import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.integrate import solve_ivp

sig = 10
B = 8/3
p = 10
# ------------------------------------------------------
# functions that returns dy/dx
# i.e. the equation we want to solve: 
# dy_j/dx = f_j(x,y_j) (j=[1,2] in this case)
def model(x,y):
    y_1 = y[0]
    y_2 = y[1]
    y_3 = y[2]
    f_1 = sig * (y_2 - y_1)
    f_2 = p*y_1 - y_2 - y_1 * y_3
    f_3 = -B * y_3 + y_1 * y_2
    return [f_1 , f_2, f_3]
# ------------------------------------------------------

# ------------------------------------------------------
# initial conditions
x0 = 0
y0_1 = 5
y0_2 = 5
y0_3 = 5
# total solution interval
t_final = 30
# step size
# not needed here. The solver solve_ivp 
# will take care of finding the appropriate step 
# ------------------------------------------------------

# ------------------------------------------------------
# Apply solve_ivp method
y = solve_ivp(model, [0 , t_final] ,[y0_1 , y0_2, y0_3])
# ------------------------------------------------------

# ------------------------------------------------------
# plot results
width = 6
height = 2
t = y.t
fig1, ax1 = plt.subplots(figsize = (width, height))
ax1.plot(t, y.y[0]) 
ax1.set_title('y1')

fig1, ax1 = plt.subplots(figsize = (width, height))
ax1.plot(t, y.y[1]) 
ax1.set_title('y2')



fig1, ax1 = plt.subplots(figsize = (width, height))
ax1.plot(t, y.y[2]) 
ax1.set_title('y3')


fig1, ax1 = plt.subplots(figsize = (width, height))
ax1.plot(y.y[0], y.y[1]) 
ax1.set_title('absciss y1,y2')

fig1, ax1 = plt.subplots(figsize = (width, height))
ax1.plot(y.y[0], y.y[2]) 
ax1.set_title('abscissa y1,y3')







##Ideal solution
# importing modules
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.integrate import solve_ivp

# ------------------------------------------------------
# functions that returns dy/dx
# i.e. the equation we want to solve: 
def model(x,y):
    sigma = 10.0
    beta =8.0/3.0
    rho = 28.0
    #rho = 10.0
    y_1 = y[0]
    y_2 = y[1]
    y_3 = y[2]
    f_1 = sigma * (y_2-y_1)
    f_2 = rho * y_1 - y_2 -y_1 * y_3
    f_3 = -beta * y_3 + y_1 * y_2
    return [f_1 , f_2, f_3]
# ------------------------------------------------------

# ------------------------------------------------------
# initial conditions
x0 = 0
y0_1 = 5
y0_2 = 5
y0_3 = 5
# total solution interval
t_final = 30
# step size
# not needed here. The solver solve_ivp 
# will take care of finding the appropriate step 
# ------------------------------------------------------

# ------------------------------------------------------
# Apply solve_ivp method
t_eval = np.linspace(0, t_final, num=5000)
y = solve_ivp(model, [0 , t_final] ,[y0_1 , y0_2, y0_3],t_eval=t_eval)
# ------------------------------------------------------

# ------------------------------------------------------
# plot results
plt.figure(1)
plt.plot(y.t,y.y[0,:] , 'b-',y.t,y.y[1,:] , 'r-',y.t,y.y[2,:] , 'g-')
plt.xlabel('t')
plt.ylabel('y_1(t), y_2(t), y_3(t)')
# ------------------------------------------------------

# ------------------------------------------------------
# plot results
plt.figure(2)
plt.plot(y.y[0,:] ,y.y[1,:],'-')
plt.xlabel('y_1')
plt.ylabel('y_2')
# ------------------------------------------------------

# ------------------------------------------------------
# plot results
plt.figure(3)
plt.plot(y.y[0,:] ,y.y[2,:],'-')
plt.xlabel('y_1')
plt.ylabel('y_3')
# ------------------------------------------------------

# ------------------------------------------------------
# print results in a text file (for later use if needed)
file_name= 'output.dat' 
f_io = open(file_name,'w') 
n_step = len(y.t)
for i in range(n_step):
    s1 = str(i)
    s2 = str(y.t[i])
    s3 = str(y.y[0,i])
    s4 = str(y.y[1,i])
    s5 = str(y.y[2,i])
    s_tot = s1 + ' ' + s2 + ' ' + s3  + ' ' + s4 + ' ' + s5
    f_io.write(s_tot + '\n')
f_io.close()
# ------------------------------------------------------

# ------------------------------------------------------
plt.show()
# ------------------------------------------------------





















##Tutorial 10
import sympy as sym
x = sym.Symbol('x')
y = sym.Symbol('y')
a = sym.Symbol('a')
c = sym.Symbol('c')
dydx = sym.Symbol('dydx')
y = c * sym.sin(a*x)
dydx = sym.diff(y,x)

print('function y(x): ', y)
print('derivative dydx: ',dydx)




import sympy as sym

# define some symbols
x = sym.Symbol('x')
y = sym.Symbol('y')

a = sym.Symbol('a')
c = sym.Symbol('c')

z = sym.Symbol('z')
s = sym.Symbol('s')
s2 = sym.Symbol('s2')
s3 = sym.Symbol('s3')

dydx = sym.Symbol('dydx')
d2ydx2 = sym.Symbol('d2ydx2')

# define a function
y=c*sym.sin(a*x)

# compute derivatives
dydx = sym.diff(y, x)
d2ydx2 = sym.diff(dydx, x)


# define another function
z = dydx + d2ydx2
s = 1/y*dydx
s2 = sym.simplify(s)
s3 = sym.series(s2, x)


# print them 
print('----------------')
print(y)
print('----------------')
print(dydx)
print('----------------')
print(d2ydx2)
print('----------------')
print(z)
print('----------------')
print(s)
print('----------------')
print(s2)
print('----------------')
print(s3)
print('----------------')