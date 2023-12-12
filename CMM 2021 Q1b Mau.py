#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 00:42:23 2023

@author: douglas
"""
import numpy as np
from scipy.integrate import quad
 
# Initial tensile force applied to the spring in Newtons.
F0 = 100  
# Spring stiffness per unit mass in reciprocal seconds squared.
s_per_m = 1500  
# Damping ratio per unit mass in reciprocal seconds.
c_over_m = 12  
 
# Spring stiffness (k) is set to s/m, m = 1
k = s_per_m  
# Damping coefficient (c) is set to c/m, m = 1.
c = c_over_m  
 
# Amplitude calculated using the spring's stiffness and the initial force.
A = F0 / k
 
# Damping constant (ωr) and angular frequency (ωi) from the system's complex roots.
ωr = 0.6230196283078494  
ωi = 24.03024141494682  
 
# Function describing the mass's displacement over time, incorporating both damping and oscillation components.
def x(t):
    return A * np.exp(-ωr * t) * np.cos(ωi * t + np.pi/8)
 
# Function for the velocity of the mass, derived as the temporal derivative of the displacement.
def v(t):
    return -A * ωr * np.exp(-ωr * t) * np.cos(ωi * t + np.pi/8) + \
           -A * ωi * np.exp(-ωr * t) * np.sin(ωi * t + np.pi/8)
 
# Damping force function, which is dependent on velocity and acts in the opposite direction.
def F(t):
    return -c * v(t)
 
# Power function representing the instantaneous rate of energy dissipation.
def P(t):
    return F(t) * v(t)
 
# Total energy dissipated by the damping force calculated by integrating the power over a specified time.
time_end = 10  # Time period for the work calculation.
work_done, _ = quad(P, 0, time_end)
 
# Output the total energy dissipated in the given time frame.
print('Total energy dissipated by the spring over 10 seconds:', work_done, 'Joules')