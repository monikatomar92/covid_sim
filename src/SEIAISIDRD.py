#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Monika Tomar
# @Date:   2020-11-29 23:56:24
# @Last Modified by:   Monika Tomar
# @Last Modified time: 2020-12-02 11:48:07

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
The SEIAISIDRD Model function
"""


def deriv_SEIAISIDRD(y, t, params):
    E, IA, IS, S, ID, R, D = y
    N = E + IA + IS + S + ID + R + D
    dS = -1 * S / N * (params["alpha"] * IA + params["beta"] * IS +
                       params["gamma"] * ID)
    dE = S / N * (params["alpha"] * IA + params["beta"] * IS +
                  params["gamma"] * ID) - params["delta"] * E
    dIA = params["epsilon"] * params["delta"] * E - params[
        "zeta"] * IA - params["netaa"] * IA
    dIS = (1 - params["epsilon"]) * params["delta"] * E - params[
        "thetas"] * IS - params["netas"] * IS - params["kappa"] * IS
    dID = params["netaa"] * IA + params["netas"] * IS - params[
        "thetad"] * ID - params["kappad"] * ID
    dR = params["zeta"] * IA + params["thetas"] * IS + params["thetad"] * ID
    dD = params["kappa"] * IS + params["kappad"] * ID

    return [dE, dIA, dIS, dS, dID, dR, dD]


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
    return (sol[:, 4:7] - gt_data).ravel()


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
R0 = 0  # Recovered
IS0 = 1
IA0 = 1
ID0 = 1
E0 = 4
D0 = 0
S0 = (N0 - IS0 - IA0 - ID0 - E0 - R0 - D0)

# S_data = (N0 - (I_data + R_data + D_data)) / N0

#*******************************************************************************

#***********************Initialization for parameters***************************
beta = 1
gamma = 0.1
alpha = 0.38
delta = 0.33
epsilon = 0.35
zeta = 1 / 14
netaa = 0.1
thetas = 0.03
netas = 0.9
kappa = 0.32
thetad = 0.039
kappad = 0.31
params = Parameters()
params.add('beta', value=beta, min=0, max=100000)
params.add('gamma', value=gamma, min=0, max=100000)
params.add('alpha', value=alpha, min=0, max=100000)
params.add('delta', value=delta, min=0, max=100000)
params.add('epsilon', value=epsilon, min=0, max=100000)
params.add('zeta', value=zeta, min=0, max=100000)
params.add('netaa', value=netaa, min=0, max=100000)
params.add('thetas', value=thetas, min=0, max=100000)
params.add('netas', value=netas, min=0, max=100000)
params.add('kappa', value=kappa, min=0, max=100000)
params.add('thetad', value=thetad, min=0, max=100000)
params.add('kappad', value=kappad, min=0, max=100000)
#*******************************************************************************

#****************************Time axis******************************************
t = np.linspace(0, 314, 314)
#*******************************************************************************

#*********************Running/Integrating the model*****************************
# **y0 = S0, E0, IA0, IS0, ID0, R0, D0
# (S, E, IA, IS, ID, R, D) = run_model(deriv_SEIAISIDRD, y0, t, args_model)
#*******************************************************************************

#*********************Estimating the parameters of the model*******************
initial_conditions = [E0, IA0, IS0, S0, ID0, R0, D0]
gt_data = (np.vstack((I_data, R_data, D_data)).T).astype(np.float64)
print("line 164 = ", gt_data.shape)
result = minimize(error,
                  params,
                  args=(initial_conditions, t, gt_data, deriv_SEIAISIDRD),
                  method="leastsq")
print(result.params)
print(params)

predicted_data = run_model(deriv_SEIAISIDRD, initial_conditions, t,
                           result.params)
print(predicted_data.shape)
# predicted_S = predicted_data[:, 3]
predicted_I = predicted_data[:, 4]
predicted_R = predicted_data[:, 5]
predicted_D = predicted_data[:, 6]
#*******************************************************************************

#**********************Plotting results*****************************************
output_file("SEIAISIDRD.html")
p = figure(plot_width=600, plot_height=600, title=None)

p.circle(range(len(I_data)), I_data, color="orange")
p.circle(range(len(R_data)), R_data, color="green")
p.circle(range(len(D_data)), D_data, color="blue")

p.cross(range(len(predicted_I)), predicted_I * N0, color="orange")
p.cross(range(len(predicted_R)), predicted_R, color="green")
p.cross(range(len(predicted_D)), predicted_D, color="blue")
show(p)
#*******************************************************************************
