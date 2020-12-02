#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Monika Tomar
# @Date:   2020-11-29 23:56:24
# @Last Modified by:   Monika Tomar
# @Last Modified time: 2020-12-02 02:35:24

import numpy as np
from scipy.integrate import odeint
import pandas as pd
from bokeh.io import output_file, show, save
from bokeh.plotting import figure
from lmfit import minimize, Parameters, Parameter, report_fit

################################################################################
#############################Data processing Related functions##################
################################################################################


def read_csv(csv_filename):
    csv_file = pd.read_csv(csv_filename)
    csv_file.set_index("Country/Region", inplace=True)
    return csv_file


def get_country_data(csv_file, country_name, start_date, end_date):
    return csv_file.loc[country_name, start_date:end_date].to_numpy()


################################################################################
#############################Model Realted functions############################
################################################################################
"""
The SIR Model function
"""


def deriv_SIR(y, t, params):
    S, I, R = y
    N = S + I + R
    dSdt = -1 * params["beta"] * S * I / N
    dIdt = params["beta"] * S * I / N - params["gamma"] * I
    dRdt = params["gamma"] * I
    return [dSdt, dIdt, dRdt]


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


def run_model(model_func, initial_conditions, time, params):
    ret = odeint(model_func, initial_conditions, time, args=(params, ))
    return ret


################################################################################
#############################Parameter Related Functions########################
################################################################################


def error(params, initial_conditions, t, gt_data, model_func):
    sol = run_model(model_func, initial_conditions, t, params)
    return (sol - gt_data).ravel()


################################################################################
#############################Running the code###################################
################################################################################

#***********************Reading data from CSVs**********************************
death_csvname = "/home/monika/COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv"
recovered_csvname = "/home/monika/COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv"
confirmed_csvname = "/home/monika/COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"

death_csv = read_csv(death_csvname)
recovered_csv = read_csv(recovered_csvname)
confirmed_csv = read_csv(confirmed_csvname)

I_data = get_country_data(confirmed_csv, "India", "1/22/20", "11/30/20")
R_data = get_country_data(recovered_csv, "India", "1/22/20", "11/30/20")
D_data = get_country_data(death_csv, "India", "1/22/20", "11/30/20")
#*******************************************************************************

#***********************Initialization for states*******************************
N0 = 1380000000
RE0 = 0  # Removed = Recovered + Dead
I0 = 1
S0 = N0 - I0 - RE0

RE_data = R_data + D_data
S_data = N0 - (I_data + R_data + D_data)
# ID0 = 1
# IA0 = 1
# IS0 = 1
# E0 = 1
# R0 = 0
# D0 = 0
# S0 = N - ID0 - IA0 - IS0 - E0 - R0 - D0
#*******************************************************************************

#***********************Initialization for parameters***************************
beta = 1
gamma = 0.1
params = Parameters()
params.add('beta', value=beta, min=0, max=10)
params.add('gamma', value=gamma, min=0, max=10)
# args_model["alpha"] = 0.38
# args_model["delta"] = 0.33
# args_model["epsilon"] = 0.35
# args_model["zeta"] = 1 / 14
# args_model["netaa"] = 0.1
# args_model["thetas"] = 0.03
# args_model["netas"] = 0.9
# args_model["kappa"] = 0.32
# args_model["thetad"] = 0.039
# args_model["kappad"] = 0.31
#*******************************************************************************

#****************************Time axis******************************************
t = np.linspace(0, 314, 314)
#*******************************************************************************

#*********************Running/Integrating the model*****************************
# **y0 = S0, E0, IA0, IS0, ID0, R0, D0
# (S, E, IA, IS, ID, R, D) = run_model(deriv_SEIAISIDRD, y0, t, args_model)
#*******************************************************************************

#*********************Estimating the parameters of the model*******************
initial_conditions = [S0, I0, RE0]
gt_data = (np.vstack((S_data, I_data, RE_data)).T).astype(np.float64)
print("line 164 = ", gt_data.shape)
result = minimize(error,
                  params,
                  args=(initial_conditions, t, gt_data, deriv_SIR),
                  method="leastsq")
print(result.params)
print(params)

predicted_data = run_model(deriv_SIR, initial_conditions, t, result.params)
print(predicted_data.shape)
predicted_S = predicted_data[:, 0]
predicted_I = predicted_data[:, 1]
predicted_R = predicted_data[:, 2]
#*******************************************************************************

#**********************Plotting results*****************************************
output_file("test.html")
p = figure(plot_width=600, plot_height=600, title=None)
p.circle(range(len(I_data)), I_data, color="orange")
p.circle(range(len(R_data)), R_data + D_data, color="green")

p.cross(range(len(predicted_I)), predicted_I, color="orange")
p.cross(range(len(predicted_R)), predicted_R, color="green")
show(p)
#*******************************************************************************
