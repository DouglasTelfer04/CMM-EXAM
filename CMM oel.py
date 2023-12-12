#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 13:22:57 2023

@author: douglas
"""


from sympy import symbols, integrate, cos, exp, pi

# Define symbols
t, A, w_r, w_i, phi, k = symbols('t A w_r w_i phi k')

# Given values
A_val = 0.1
w_r_val = -0.62
w_i_val = 24.03
phi_val = pi / 8
k_val = 100 / A_val  # Spring constant calculated from initial force of 100 N

# Define displacement function
x = A * exp(w_r * t) * cos(w_i * t + phi)

# Define force function
force = -k * x

# Integrate force with respect to displacement over the interval [0, 10]
work_done_func = integrate(force, (t, 0, 10))
work_done = work_done_func.subs({A: A_val, w_r: w_r_val, w_i: w_i_val, phi: phi_val, k: k_val})

print("The work done in the first 10 seconds is:", work_done.evalf(), "Joules")

'''
method is:
    
integrate in sympy, then sub in values into integral.

The work done on an object is given by the integral of the force with respect to 
displacement. In the case of a spring, the force exerted by the spring is proportional
to the displacement. The equation for the force exerted by the spring is given by Hooke's Law:

F(t)= −k⋅x(t)

where 
k is the spring constant. 

x(t)=A⋅e(w_rt)*cos(w_i*t+phi)

so the force is:

F(t)= −k⋅A⋅e(w_rt)*cos(w_i*t+phi)
the work done is the integral of the force with respect to displacement over the specified time interval. The work 

W is given by:

W = int(0,10) F(t).dx

Using the given values, we can calculate this integral to find the work done.
'''
