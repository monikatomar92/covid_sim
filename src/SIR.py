#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Monika Tomar
# @Date:   2020-11-29 23:56:24
# @Last Modified by:   Monika Tomar
# @Last Modified time: 2020-11-30 15:12:30

import numpy as np
from scipy.integrate import odeint

################################################################################
#############################Model Realted functions############################
################################################################################
"""
The SIR Model function
"""


def deriv_SIR(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt


"""
The SEIAISIDRD Model function
"""


def deriv_SEIAISIDRD(y, t, args_model):
    S, E, IA, IS, ID, R, D = y

    dS = -1 * S * (args_model["alpha"] * IA + args_model["beta"] * IS +
                   args_model["gamma"] * ID)
    dE = S * (args_model["alpha"] * IA + args_model["beta"] * IS +
              args_model["gamma"] * ID) - args_model["delta"] * E
    dIA = args_model["epsilon"] * args_model["delta"] * E - args_model[
        "zeta"] * IA - args_model["netaa"] * IA
    dIS = (1 - args_model["epsilon"]) * args_model["delta"] * E - args_model[
        "thetas"] * IS - args_model["netas"] * IS - args_model["kappa"] * IS
    dID = args_model["netaa"] * IA + args_model["netas"] * IS - args_model[
        "thetad"] * ID - args_model["kappad"] * ID
    dR = args_model["zeta"] * IA + args_model["thetas"] * IS + args_model[
        "thetad"] * ID
    dD = args_model["kappa"] * IS + args_model["kappad"] * ID

    return dS, dE, dIA, dIS, dID, dR, dD


"""
Runs the model using initial conditions and a model function and integrates it to time t
"""


def run_model(model_func, initial_conditions, time, args_model):
    ret = odeint(model_func, initial_conditions, time, args=(args_model, ))
    return ret.T


################################################################################
#############################Running the code###################################
################################################################################

#***********************Initialization for states*******************************
N = 1000
ID0 = 1
IA0 = 1
IS0 = 1
E0 = 1
R0 = 0
D0 = 0
S0 = N - ID0 - IA0 - IS0 - E0 - R0 - D0
#*******************************************************************************

#***********************Initialization for parameters***************************
args_model = {}
args_model["alpha"] = 0.38
args_model["beta"] = 0.16
args_model["gamma"] = 0.09
args_model["delta"] = 0.33
args_model["epsilon"] = 0.35
args_model["zeta"] = 1 / 14
args_model["netaa"] = 0.1
args_model["thetas"] = 0.03
args_model["netas"] = 0.9
args_model["kappa"] = 0.32
args_model["thetad"] = 0.039
args_model["kappad"] = 0.31
#*******************************************************************************

#****************************Time axis******************************************
t = np.linspace(0, 160, 160)
#*******************************************************************************

#*********************Running/Integrating the model*****************************
y0 = S0, E0, IA0, IS0, ID0, R0, D0
(S, E, IA, IS, ID, R, D) = run_model(deriv_SEIAISIDRD, y0, t, args_model)
#*******************************************************************************
