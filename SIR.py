#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Monika Tomar
# @Date:   2020-11-29 23:56:24
# @Last Modified by:   Monika Tomar
# @Last Modified time: 2020-11-30 15:12:30

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Initialization for states
N = 1000
ID0 = 1
IA0 = 1
IS0 = 1
E0 = 1
R0 = 0
D0 = 0
S0 = N - ID0 - IA0 - IS0 - E0 - R0 - D0

# Initialization for parameters
alpha = 0.38
beta = 0.16
gamma = 0.09
delta = 0.33
epsilon = 0.35
zeta = 1 / 14
netaa = 0.1
thetas = 0.03
netas = 0.9
kappa = 0.32
thetad = 0.039
kappad = 0.31
# Time axis
t = np.linspace(0, 160, 160)

# t = np.array([160])


# The SIR model differential equations.
def deriv_SIR(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt


def deriv_SEIAISIDRD(y, t, alpha, beta, gamma, delta, epsilon, zeta, netaa,
                     thetas, netas, kappa, thetad, kappad):
    S, E, IA, IS, ID, R, D = y
    dS = -1 * S * (alpha * IA + beta * IS + gamma * ID)
    dE = S * (alpha * IA + beta * IS + gamma * ID) - delta * E
    dIA = epsilon * delta * E - zeta * IA - netaa * IA
    dIS = (1 - epsilon) * delta * E - thetas * IS - netas * IS - kappa * IS
    dID = netaa * IA + netas * IS - thetad * ID - kappad * ID
    dR = zeta * IA + thetas * IS + thetad * ID
    dD = kappa * IS + kappad * ID

    return dS, dE, dIA, dIS, dID, dR, dD


# Initial conditions vector
y0 = S0, E0, IA0, IS0, ID0, R0, D0
# Integrate the SIR equations over the time grid, t.
# ret = odeint(deriv, y0, t, args=(N, beta, gamma))
ret = odeint(deriv_SEIAISIDRD,
             y0,
             t,
             args=(alpha, beta, gamma, delta, epsilon, zeta, netaa, thetas,
                   netas, kappa, thetad, kappad))
S, E, IA, IS, ID, R, D = ret.T

# print(S0, I0, R0)
# print(S, S.shape)
# print(I, I.shape)
# print(R, R.shape)

print(S)
# print(E[-1])
# print(IA)
# print(IS)
# print(ID)
# print(R)
print(D)
# print(S[-1] + E[-1] + IA[-1] + IS[-1] + ID[-1] + R[-1] + D[-1])

# Plot the data on three separate curves for S(t), I(t) and R(t)
# fig = plt.figure(facecolor='w')
# ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
# ax.plot(t, S / 1000, 'b', alpha=0.5, lw=2, label='Susceptible')
# ax.plot(t, ID / 1000, 'r', alpha=0.5, lw=2, label='Infected')
# ax.plot(t, R / 1000, 'g', alpha=0.5, lw=2, label='Recovered with immunity')
# ax.set_xlabel('Time /days')
# ax.set_ylabel('Number (1000s)')
# ax.set_ylim(0, 1.2)
# ax.yaxis.set_tick_params(length=0)
# ax.xaxis.set_tick_params(length=0)
# ax.grid(b=True, which='major', c='w', lw=2, ls='-')
# legend = ax.legend()
# legend.get_frame().set_alpha(0.5)
# for spine in ('top', 'right', 'bottom', 'left'):
#     ax.spines[spine].set_visible(False)
# plt.show()