#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 23:35:44 2023

@author: douglas
"""
##QUESTION 1


import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Define the differential equation
def model(y, x, w, T):
    dydx = [y[1], (w/T) * np.sqrt(1 + y[1]**2)]
    return dydx

# Initial conditions
y0 = [0, 0]  # Initial values for y and dy/dx
x = np.linspace(0, 10, 100)  # Define the x values

# Parameters
w = 1.0
T = 2.0

# Solve the differential equation using odeint
solution = odeint(model, y0, x, args=(w, T))

# Extract y values from the solution
y_values = solution[:, 0]

# Plot the solution
plt.plot(x, y_values, label='y(x)')
plt.xlabel('x')
plt.ylabel('y(x)')
plt.legend()
plt.show()


import numpy as np
from scipy.optimize import root

# Define the equation as a function
def equation_to_solve(T):
    return T/10 * np.cosh(10/T * 50) + 5 - T/10 - 15

# Initial guess for the solution
initial_guess = 3

# Use root with 'hybr' method to find the numerical solution
result = root(equation_to_solve, initial_guess, method='hybr')

# Display the result
numerical_solution = result.x[0]
print("Numerical Solution for T:", numerical_solution)








##QUESTION 2 LONG

# importing modules


import numpy as np
import matplotlib.pyplot as plt
import math

# ------------------------------------------------------
# functions that returns dy/dx
# i.e. the equation we want to solve: dy_j/dx = f_j(x,y_j) (j=[1,2] in this case)
def model(t,w,theta):
    f_1 = w
    f_2 = -np.sin(theta)
    return [f_1 , f_2]
# ------------------------------------------------------


# ------------------------------------------------------
# initial conditions
t0 = 0
w0 = 0
theta0 = np.pi/4
# total solution interval
t_final = 40
# step size
h = 0.001
# ------------------------------------------------------


# ------------------------------------------------------
# Euler method

# number of steps
n_step = math.ceil(t_final/h)

# Definition of arrays to store the solution
w_eul = np.zeros(n_step+1)
theta_eul = np.zeros(n_step+1)
t_eul = np.zeros(n_step+1)

# Initialize first element of solution arrays 
# with initial condition
w_eul[0] = w0
theta_eul[0] = theta0
t_eul[0]   = t0 

# Populate the x array
for i in range(n_step):
    t_eul[i+1]  = t_eul[i]  + h

# Apply Euler method n_step times
for i in range(n_step):
    # compute the slope using the differential equation
    [slope_1 , slope_2] = model(t_eul[i],w_eul[i],theta_eul[i]) 
    # use the Euler method
    w_eul[i+1] = w_eul[i] + h * slope_1
    theta_eul[i+1] = theta_eul[i] + h * slope_2
    'print(w_eul[i],theta_eul[i])'
# ------------------------------------------------------
times_of_interest = [10, 20, 30]
for time_point in times_of_interest:
    index_at_time_point = int(time_point / h)
    print(f"At t = {time_point}: theta = {theta_eul[index_at_time_point]}")
times_of_interest = [10, 20, 30]
for time_point in times_of_interest:
    index_at_time_point = int(time_point / h)
    print(f"At t = {time_point}: w = {w_eul[index_at_time_point]}")

# ------------------------------------------------------
# plot results
plt.plot(t_eul, w_eul , 'b.-')
plt.xlabel('x')
plt.ylabel('y_1(x), y_2(x)')
plt.show()
# ------------------------------------------------------
plt.plot(t_eul, theta_eul , 'r.-')
plt.xlabel('x')
plt.ylabel('y_1(x), y_2(x)')
plt.show()



import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.signal import find_peaks

# ------------------------------------------------------
# functions that returns dy/dx
# i.e. the equation we want to solve: dy_j/dx = f_j(x,y_j) (j=[1,2] in this case)
def model(t, w, theta):
    f_1 = w
    f_2 = -np.sin(theta)
    return [f_1, f_2]

# ------------------------------------------------------
# Function to find the period
def find_period(times, values):
    peaks, _ = find_peaks(values)
    periods = np.diff(times[peaks])
    return np.mean(periods)

# ------------------------------------------------------
# initial conditions
t0 = 0
w0 = 0
theta0 = np.pi/2
# total solution interval
t_final = 40
# step size
h = 0.001
# ------------------------------------------------------

# ------------------------------------------------------
# Euler method

# number of steps
n_step = math.ceil(t_final/h)

# Definition of arrays to store the solution
w_eul = np.zeros(n_step+1)
theta_eul = np.zeros(n_step+1)
t_eul = np.zeros(n_step+1)

# Initialize first element of solution arrays 
# with initial condition
w_eul[0] = w0
theta_eul[0] = theta0
t_eul[0] = t0

# Populate the t array
for i in range(n_step):
    t_eul[i+1] = t_eul[i] + h

# Apply Euler method n_step times
for i in range(n_step):
    # compute the slope using the differential equation
    [slope_1, slope_2] = model(t_eul[i], w_eul[i], theta_eul[i]) 
    # use the Euler method
    w_eul[i+1] = w_eul[i] + h * slope_2
    theta_eul[i+1] = theta_eul[i] + h * slope_1

# Find the period
period = find_period(t_eul, theta_eul)

# Print the period
print(f"Estimated period: {period:.2f} seconds")

# ------------------------------------------------------
# plot results
plt.plot(t_eul, theta_eul, 'r.-')
plt.xlabel('Time')
plt.ylabel('Theta')
plt.title('Oscillation of Theta vs. Time')
plt.show()








import numpy as np
import matplotlib.pyplot as plt
import math
# ------------------------------------------------------
# Function that returns dy/dx for the given differential equations
def model(theta, omega):
    g = 9.8  # acceleration due to gravity
    l = 1.0  # length of the pendulum
    dtheta_dt = omega
    domega_dt = -g / l * np.sin(theta)
    return [dtheta_dt, domega_dt]
# ------------------------------------------------------
# Initial conditions
theta0 = np.pi / 4  # initial angle
omega0 = 0.0        # initial angular velocity
# ------------------------------------------------------
# Total solution interval
t_final = 20
# Step size
h = 0.001
# ------------------------------------------------------
# Euler method
n_steps = math.ceil(t_final / h)
# ------------------------------------------------------
# Arrays to store the solution
theta_euler = np.zeros(n_steps + 1)
omega_euler = np.zeros(n_steps + 1)
t_euler = np.zeros(n_steps + 1)
# ------------------------------------------------------
# Initialize first element of solution arrays with initial conditions
theta_euler[0] = theta0
omega_euler[0] = omega0
t_euler[0] = 0.0
# ------------------------------------------------------
# Apply Euler method n_steps times
for i in range(n_steps):
    # Compute the slope using the differential equation
    [slope_theta, slope_omega] = model(theta_euler[i], omega_euler[i])
    # Use the Euler method
    theta_euler[i + 1] = theta_euler[i] + h * slope_theta
    omega_euler[i + 1] = omega_euler[i] + h * slope_omega
    t_euler[i + 1] = t_euler[i] + h
# ------------------------------------------------------
# Plot results
plt.plot(t_euler, theta_euler, 'b.-', label='Theta(t)')
plt.plot(t_euler, omega_euler, 'r.-', label='Omega(t)')
plt.xlabel('Time (t)')
plt.ylabel('Theta(t), Omega(t)')
plt.legend()
plt.show()
# ------------------------------------------------------
#Print the value of y(x) at x = 10
t_at_10_index = np.abs(t_euler - 10).argmin()
theta_at_10 = theta_euler[t_at_10_index]
omega_at_10 = omega_euler[t_at_10_index]
print("omega(10) =", omega_at_10)
print("theta(10) =", theta_at_10)
# ------------------------------------------------------
#Print the value of y(x) at x = 20
t_at_20_index = np.abs(t_euler - 20).argmin()
theta_at_20 = theta_euler[t_at_20_index]
omega_at_20 = omega_euler[t_at_20_index]
print("omega(20) =", omega_at_20)
print("theta(20) =", theta_at_20)
# ------------------------------------------------------
#Print the value of y(x) at x = 30
t_at_30_index = np.abs(t_euler - 30).argmin()
theta_at_30 = theta_euler[t_at_30_index]
omega_at_30 = omega_euler[t_at_30_index]
print("omega(30) =", omega_at_30)
print("theta(30) =", theta_at_30)

# ------------------------------------------------------





























##QUESTION 3 Interest rates




import numpy as np
from scipy.optimize import fsolve

# Define the equation as a Python function
def equation_to_solve(x):
    return 115000 * (x * (1 + x)**6)/((1 + x)**6 - 1) - 25500

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
f = lambda x: 115000 * (x * (1 + x)**6)/((1 + x)**6 - 1) - 25500
approx_phi = bisection(f,0.001,0.8,50)
print(f'bisection approx: {approx_phi}')







##QUESTION 4
t = 30
u = 1.8E3
m0 = 160E3
q = 2.5E3

s = (u * np.log(m0/(m0 - q * t)) * t) + 1/2 * t**2 * u *(1/(64 - t))
print(s)
##initially suvat was tried but easier to just integrate velocity equation
from scipy.integrate import quad
def integrand(t, u, m0, q):
    return u * np.log(m0/(m0 - q * t))


I = quad(integrand, 0, 30, args=(u, m0, q))
print(I)












print("numerical integration")

import numpy as np

def func(x): 
      
    return 1800 * np.log((160000/(160000 - 2500 * x)))
  
# Function to perform calculations 
def calculate(lower_limit, upper_limit, interval_limit ): 
      
    interval_size = (float(upper_limit - lower_limit) / interval_limit) 
    sum = func(lower_limit) + func(upper_limit); 
   
    # Calculates value till integral limit 
    for i in range(1, interval_limit ): 
        if (i % 3 == 0): 
            sum = sum + 2 * func(lower_limit + i * interval_size) 
        else: 
            sum = sum + 3 * func(lower_limit + i * interval_size) 
      
    return ((float( 3 * interval_size) / 8 ) * sum ) 
  
# driver function 
interval_limit = 10000
lower_limit = 0
upper_limit = 30
  
integral_res = calculate(lower_limit, upper_limit, interval_limit) 
  
# rounding the final answer to 6 decimal places  
print (round(integral_res, 6)) 
  
# This code is contributed by Saloni. 

  
integral_res = calculate(lower_limit, upper_limit, interval_limit) 
  
# rounding the final answer to 6 decimal places  
print (round(integral_res, 6)) 



#integration, a technique that approximates the definite integral of a function over an interval. 
#The fundamental idea of Simpson's rule is to divide the interval into smaller subintervals and then 
#use quadratic polynomials to approximate the function within each subinterval. The 3/8 Simpson's rule
# specifically employs quadratic approximations for three consecutive subintervals. It calculates the 
#integral by evaluating the function at the endpoints and intermediate points within each subinterval, 
#assigning different weights (coefficients of 3 and 2) based on the position of the evaluation point. 
#These weighted function values are then summed up over all subintervals, and the result is scaled by 
#a factor of 3/8 to obtain the final numerical estimate of the integral. The choice of the 3/8 rule 
#allows for more accurate approximations compared to the standard 1/3 Simpson's rule in certain cases.
# The code iterates over the specified subintervals, applies the 3/8 Simpson's rule formula, and
# accumulates the results to provide an approximate solution to the definite integral of the given function.


















##QUESTION 5

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Constants
tank_radius = 1
tank_cross_sectional_area = 0.01
gravitational_acceleration = 9.81
initial_water_height = 1

# Bernoulli's equation for flow rate Q
calculate_flow_rate = lambda water_height: tank_cross_sectional_area * np.sqrt(2 * gravitational_acceleration * water_height)

# Differential equation for the cylindrical tank
def cylindrical_tank_dh_dt(t, water_height):
    return -calculate_flow_rate(water_height) / (np.pi * tank_radius**2)

# Solve the differential equation for the cylindrical tank
time_span = [0, 10000]
initial_conditions = [initial_water_height]
solution_cylindrical = solve_ivp(cylindrical_tank_dh_dt, time_span, initial_conditions, method='RK45', max_step=0.1)

# Time to drain the cylindrical tank
t_drain_cylindrical = solution_cylindrical.t[-1]
print(f'Time to drain the cylindrical tank: {t_drain_cylindrical} seconds')

# Plot the height of water over time for the cylindrical tank
plt.figure(figsize=(10, 6))
plt.plot(solution_cylindrical.t, solution_cylindrical.y[0])
plt.title('Water Height vs. Time for Cylindrical Tank')
plt.xlabel('Time (seconds)')
plt.ylabel('Water Height (meters)')
plt.grid(True)
plt.show()

# Function to calculate the radius at a given height z
initial_slope_guess = 1.12
calculate_radius_at_height = lambda height: tank_radius + initial_slope_guess * height

# Differential equation for the truncated cone-shaped tank
def cone_tank_dh_dt(t, water_height):
    return -calculate_flow_rate(water_height) / (np.pi * calculate_radius_at_height(water_height)**2)

# Solve the differential equation for the truncated cone-shaped tank with initial guess for slope
solution_cone = solve_ivp(cone_tank_dh_dt, time_span, initial_conditions, method='RK45', max_step=0.1)

# Time to drain the truncated cone-shaped tank with initial guess for slope
t_drain_cone = solution_cone.t[-1]
print(f'Time to drain the truncated cone-shaped tank with initial guess for slope s = {initial_slope_guess}: {t_drain_cone} seconds')

# Function to find the slope s that doubles the drain time
def find_optimal_slope(initial_slope, target_time):
    for _ in range(1000):
        calculate_radius_at_height = lambda height: tank_radius + initial_slope * height
        cone_tank_dh_dt = lambda t, h: -calculate_flow_rate(h) / (np.pi * calculate_radius_at_height(h)**2)
        solution_cone = solve_ivp(cone_tank_dh_dt, time_span, initial_conditions, method='RK45', max_step=0.01)
        t_drain_cone = solution_cone.t[-1]
        
        if np.isclose(t_drain_cone, target_time, rtol=1e-2):  # 1% tolerance
            return initial_slope
        elif t_drain_cone < target_time:
            initial_slope += 0.001
        else:
            initial_slope -= 0.001

    return initial_slope

# Target time is double the cylindrical tank drain time
target_time_double_cylindrical = 2 * t_drain_cylindrical
optimal_slope = find_optimal_slope(initial_slope_guess, target_time_double_cylindrical)

print(f'Optimal slope s to double the drain time: {optimal_slope}')







##For this specific problem, the differential equations governing water height in cylindrical
# and truncated cone-shaped tanks are complex and involve square roots and changing geometries.
# solve_ivp is well-suited for handling such intricacies and provides a convenient interface for
# specifying the ODEs, initial conditions, and integration parameters.

#In summary, solve_ivp was chosen for its robustness, adaptability, and efficiency in solving the
# system of ODEs characterizing the water drainage in different tank geometries. Its ability to
# dynamically adjust step sizes and handle diverse ODE systems makes it a suitable choice for this
# simulation.

#This code defines a function find_slope that iteratively adjusts the slope of the truncated cone-shaped
# tank until its drain time is approximately double that of the cylindrical tank. The function uses t
#he solve_ivp function to solve the differential equation for the truncated cone-shaped tank, and the
# iterative approach ensures convergence to the optimal slope. The target time is set to be double
# the drain time of the cylindrical tank, and the optimal slope is printed at the end.

