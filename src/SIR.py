#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Monika Tomar
# @Date:   2020-11-29 23:56:24
# @Last Modified by:   Monika Tomar
# @Last Modified time: 2020-12-04 15:11:41

import numpy as np
from scipy.integrate import odeint
import pandas as pd
from bokeh.io import output_file, show, save
from bokeh.layouts import gridplot
from bokeh.models import Div
from bokeh.plotting import figure
from bokeh.layouts import column, layout
from bokeh.models.widgets import Panel, Tabs
from lmfit import minimize, Parameters, Parameter, report_fit

################################################################################
#############################Data processing Related functions##################
################################################################################


def read_csv(csv_filename):
    csv_file = pd.read_csv(csv_filename)
    csv_file.set_index("Country/Region", inplace=True)
    return csv_file


def get_country_data(csv_file, country_name, start_date, n_days):
    end_date_pd = pd.to_datetime(start_date) + pd.DateOffset(days=n_days - 1)
    end_date = end_date_pd.strftime("%-m/%-d/%y")
    return csv_file.loc[country_name, start_date:end_date].to_numpy()


################################################################################
#############################Model Related functions############################
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
Runs the model using initial conditions and a model function and integrates it to time t
"""


def run_model(model_func, initial_conditions, time, params):
    ret = odeint(model_func, initial_conditions, time, args=(params, ))
    return ret


def predict_SIR(model_func, initial_conditions, time, params):
    predicted_data = run_model(model_func, initial_conditions, time, params)
    predicted_S = predicted_data[:, 0]
    predicted_I = predicted_data[:, 1]
    predicted_R = predicted_data[:, 2]
    return predicted_S, predicted_I, predicted_R


################################################################################
#############################Parameter Related Functions########################
################################################################################


def error(params, initial_conditions, t, gt_data, model_func):
    sol = run_model(model_func, initial_conditions, t, params)
    return (sol - gt_data).ravel()


def estimate_params(confirmed_csv, recovered_csv, death_csv, population_dict,
                    country, start_date, n_simulation, n_prediction):
    I_data = get_country_data(confirmed_csv, country, start_date, n_simulation)
    R_data = get_country_data(recovered_csv, country, start_date, n_simulation)
    D_data = get_country_data(death_csv, country, start_date, n_simulation)
    #*******************************************************************************

    #***********************Initialization for states*******************************
    N0 = population_dict[country]
    RE0 = 0
    I0 = 1
    S0 = N0 - I0 - RE0

    RE_data = R_data + D_data
    S_data = N0 - (I_data + R_data + D_data)
    #*******************************************************************************

    #***********************Initialization for parameters***************************
    beta = 1
    gamma = 0.1
    params = Parameters()
    params.add('beta', value=beta, min=0, max=10)
    params.add('gamma', value=gamma, min=0, max=10)
    #*******************************************************************************

    #****************************Time axis******************************************
    t = np.linspace(0, n_simulation, n_simulation)
    t_predict = np.linspace(0, n_prediction, n_prediction)
    #*******************************************************************************

    #*********************Estimating the parameters of the model*******************
    initial_conditions = [S0, I0, RE0]
    gt_data = (np.vstack((S_data, I_data, RE_data)).T).astype(np.float64)
    result = minimize(error,
                      params,
                      args=(initial_conditions, t, gt_data, deriv_SIR),
                      method="leastsq")

    return S_data, I_data, RE_data, result.params, initial_conditions, t_predict


################################################################################
#############################Visualization functions############################
################################################################################


def visualize(country, n_days, S, I, R, S_p, I_p, R_p):
    div = Div(text="SIR Model",
              width=200,
              height=20,
              align="center",
              style={
                  'font-size': '200%',
                  'font-weight': 'bold'
              })
    tools = "hover,box_select,pan,xwheel_zoom,xbox_zoom,save,reset"
    tooltips = [("Number of people", "@y{int}"), ("Days", "@x")]
    net_layout = []
    for c in country:
        country_layout = []
        for n in n_days:
            key = c + "," + str(n)
            title = "Country:" + c + "\nNumber of days:" + str(n)
            p = figure(plot_width=450,
                       plot_height=280,
                       title=title,
                       tools=tools,
                       tooltips=tooltips)
            p.xaxis.axis_label = 'Number of days'
            p.xaxis.formatter.use_scientific = False
            p.yaxis.axis_label = 'Number of people'
            p.yaxis.formatter.use_scientific = False

            p.circle(range(len(I[key])),
                     I[key],
                     color="orange",
                     legend_label="I")
            p.circle(range(len(R[key])),
                     R[key],
                     color="green",
                     legend_label="R")
            p.line(range(len(I[key])), I[key], color="orange", alpha=0.5)
            p.line(range(len(R[key])), R[key], color="green", alpha=0.5)

            p.square(range(len(I_p[key])),
                     I_p[key],
                     color="blue",
                     legend_label="Predicted I")
            p.square(range(len(R_p[key])),
                     R_p[key],
                     color="purple",
                     legend_label="Predicted R")
            p.line(range(len(I_p[key])), I_p[key], color="blue", alpha=0.5)
            p.line(range(len(R_p[key])), R_p[key], color="purple", alpha=0.5)
            p.legend.location = "top_left"
            p.legend.label_text_font_size = "7pt"
            country_layout.append(p)
        net_layout.append(country_layout)
    net_fig = gridplot(net_layout)
    net_layout = column(div, net_fig)
    return Panel(child=net_layout, title="Model Expressivity")


def visualize_600(country, n_days, initial_conditions):
    color_dict = {75: "blue", 150: "green", 225: "red", 300: "purple"}
    t_600 = np.linspace(0, 600, 600)
    tools = "hover,box_select,pan,xwheel_zoom,xbox_zoom,save,reset"
    tooltips = [("Number of people", "@y{int}"), ("Days", "@x")]
    net_600_layout = []
    for c in country:
        p = figure(
            plot_width=600,
            plot_height=800,
            title="Relative predictions using estimated parameters for " + c,
            tools=tools,
            tooltips=tooltips)
        p.xaxis.axis_label = 'Number of days'
        p.xaxis.formatter.use_scientific = False
        p.yaxis.axis_label = 'Number of people'
        p.yaxis.formatter.use_scientific = False

        for n in n_days:
            key = c + "," + str(n)
            S_p_600, I_p_600, R_p_600 = predict_SIR(deriv_SIR,
                                                    initial_conditions, t_600,
                                                    params[key])
            p.cross(range(len(I_p_600)),
                    I_p_600,
                    color=color_dict[n],
                    legend_label="I for t=" + str(n))
            p.line(range(len(I_p_600)),
                   I_p_600,
                   color=color_dict[n],
                   alpha=0.5)
            p.circle(range(len(R_p_600)),
                     R_p_600,
                     color=color_dict[n],
                     legend_label="R for t=" + str(n))
            p.line(range(len(R_p_600)),
                   R_p_600,
                   color=color_dict[n],
                   alpha=0.5)
        p.legend.location = "top_left"
        p.legend.label_text_font_size = "7pt"
        net_600_layout.append(p)
    net_fig = layout([net_600_layout])
    return Panel(child=net_fig, title="Parameter Prediction")


def visualize_parameter_sweep(params, initial_conditions, t):
    country_sweep = "India"
    n_days_sweep = 75
    key = country_sweep + "," + str(n_days_sweep)
    params_sweep = params[key]
    S_p, I_p, R_p = predict_SIR(deriv_SIR, initial_conditions, t, params_sweep)
    params_sweep["beta"].value = (params_sweep["beta"].value) * 2
    S_p_beta, I_p_beta, R_p_beta = predict_SIR(deriv_SIR, initial_conditions,
                                               t, params_sweep)

    tools = "hover,box_select,pan,xwheel_zoom,xbox_zoom,save,reset"
    tooltips = [("Number of people", "@y{int}"), ("Days", "@x")]
    p1 = figure(
        plot_width=800,
        plot_height=600,
        title=
        "Adjusting for Compliance/Non-compliance,For asyptomatic and superspreader infections",
        tools=tools,
        tooltips=tooltips)
    p1.xaxis.axis_label = 'Number of days'
    p1.xaxis.formatter.use_scientific = False
    p1.yaxis.axis_label = 'Number of people'
    p1.yaxis.formatter.use_scientific = False

    p1.circle(range(len(I_p)),
              I_p,
              color="orange",
              legend_label="I, lower beta")
    p1.circle(range(len(R_p)),
              R_p,
              color="green",
              legend_label="R, lower beta")
    p1.line(range(len(I_p)), I_p, color="orange", alpha=0.5)
    p1.line(range(len(R_p)), R_p, color="green", alpha=0.5)

    p1.circle(range(len(I_p_beta)),
              I_p_beta,
              color="blue",
              legend_label="I, higher beta")
    p1.circle(range(len(R_p_beta)),
              R_p_beta,
              color="purple",
              legend_label="R, higher beta")
    p1.line(range(len(I_p_beta)), I_p_beta, color="blue", alpha=0.5)
    p1.line(range(len(R_p_beta)), R_p_beta, color="purple", alpha=0.5)

    params_sweep["beta"].value = (params_sweep["beta"].value) * 0.5
    params_sweep["gamma"].value = (params_sweep["gamma"].value) * 0.5
    S_p_gamma, I_p_gamma, R_p_gamma = predict_SIR(deriv_SIR,
                                                  initial_conditions, t,
                                                  params_sweep)

    p2 = figure(plot_width=800,
                plot_height=600,
                title="Adjusting for infection load",
                tools=tools,
                tooltips=tooltips)
    p2.xaxis.axis_label = 'Number of days'
    p2.xaxis.formatter.use_scientific = False
    p2.yaxis.axis_label = 'Number of people'
    p2.yaxis.formatter.use_scientific = False

    p2.circle(range(len(I_p)),
              I_p,
              color="orange",
              legend_label="I, higher gamma")
    p2.circle(range(len(R_p)),
              R_p,
              color="green",
              legend_label="R, higher gamma")
    p2.line(range(len(I_p)), I_p, color="orange", alpha=0.5)
    p2.line(range(len(R_p)), R_p, color="green", alpha=0.5)

    p2.circle(range(len(I_p_gamma)),
              I_p_gamma,
              color="blue",
              legend_label="I, lower gamma")
    p2.circle(range(len(R_p_gamma)),
              R_p_gamma,
              color="purple",
              legend_label="R, lower gamma")
    p2.line(range(len(I_p_gamma)), I_p_gamma, color="blue", alpha=0.5)
    p2.line(range(len(R_p_gamma)), R_p_gamma, color="purple", alpha=0.5)

    net_fig = layout([[p1, p2]])
    return Panel(child=net_fig, title="Parameter Sweep")


################################################################################
#############################Running the code###################################
################################################################################

#***********************Reading data from CSVs**********************************
population_dict = {
    "India": 1380000000,
    "Italy": 60360000,
    "New Zealand": 4886000,
    "Brazil": 210000000
}
start_date = "1/22/20"

confirmed_csvname = "/home/monika/COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"
recovered_csvname = "/home/monika/COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv"
death_csvname = "/home/monika/COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv"

confirmed_csv = read_csv(confirmed_csvname)
recovered_csv = read_csv(recovered_csvname)
death_csv = read_csv(death_csvname)
#*******************************************************************************
#***********************Modelling and Prediction********************************
S = {}
I = {}
R = {}
S_p = {}
I_p = {}
R_p = {}
params = {}

country = ["India", "Italy", "New Zealand"]
n_days = [75, 150, 225, 300]

t = None
initial_conditions = None
for c in country:
    for n in n_days:
        key = c + "," + str(n)
        S[key], I[key], R[key], params[
            key], initial_conditions, t = estimate_params(
                confirmed_csv, recovered_csv, death_csv, population_dict, c,
                start_date, min(n, 314), n)
        S_p[key], I_p[key], R_p[key] = predict_SIR(deriv_SIR,
                                                   initial_conditions, t,
                                                   params[key])
#*******************************************************************************
#***********************Visualization*******************************************
output_file("SIR_dashboard.html")

display_tabs = []
display_tabs.append(visualize(country, n_days, S, I, R, S_p, I_p, R_p))
display_tabs.append(visualize_600(country, n_days, initial_conditions))
display_tabs.append(visualize_parameter_sweep(params, initial_conditions, t))

dashboard = Tabs(tabs=display_tabs)
show(dashboard)
#*******************************************************************************

################################################################################
#############################End################################################
################################################################################