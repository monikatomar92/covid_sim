#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Monika Tomar
# @Date:   2020-11-29 23:56:24
# @Last Modified by:   Monika Tomar
# @Last Modified time: 2020-12-06 23:24:52

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
import pickle

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


def deriv_own_model(y, t, params):
    Sc, Snc, Ell, Ehl, Ia, Is, Iss, Qh, Qhos, R, D = y
    N = Sc + Snc + Ell + Ehl + Ia + Is + Iss + Qh + Qhos + R + D

    Sc_rate = (Sc / N) * (params["beta_a"] * Ia + params["beta_s"] * Is +
                          params["beta_Qhos"] * Qhos + params["beta_Qh"] * Qh)
    Snc_rate = (Snc / N) * (params["beta_a"] * Ia + params["beta_s"] * Is +
                            params["beta_ss"] * Iss + params["beta_Qhos"] *
                            Qhos + params["beta_Qh"] * Qh)
    dScdt = -1 * Sc_rate + params["alpha_c"] * Snc - params[
        "alpha_nc"] * Sc + params["theta"] * params["rho"] * R
    dSncdt = -1 * Snc_rate + params["alpha_nc"] * Sc - params[
        "alpha_c"] * Snc + params["theta"] * (1 - params["rho"]) * R

    dElldt = params["gamma_llc"] * Sc_rate + params[
        "gamma_llnc"] * Snc_rate - params["kappa_ll"] * Ell
    dEhldt = (1 - params["gamma_llc"]) * Sc_rate + (
        1 - params["gamma_llnc"]) * Snc_rate - params["kappa_hl"] * Ehl

    dIadt = params["kappa_ll"] * params["rho_lla"] * Ell + params[
        "kappa_hl"] * params["rho_hla"] * Ehl - params["delta_a"] * Ia
    dIsdt = params["kappa_ll"] * params["rho_lls"] * Ell + params[
        "kappa_hl"] * params["rho_hls"] * Ehl - params["delta_s"] * Is
    dIssdt = params["kappa_ll"] * (
        1 - params["rho_lla"] - params["rho_lls"]
    ) * Ell + params["kappa_hl"] * (1 - params["rho_hla"] - params["rho_hls"]
                                    ) * Ehl - params["delta_ss"] * Iss

    dQhdt = params["delta_a"] * Ia + params["delta_s"] * (
        1 - params["ita_s"]) * Is + params["delta_ss"] * (
            1 - params["ita_ss"]
        ) * Iss - params["theta_h"] * Qh - params["lambda_h"] * Qh
    dQhosdt = params["delta_s"] * params["ita_s"] * Is + params[
        "delta_ss"] * params["ita_ss"] * Iss - params[
            "theta_hos"] * Qhos - params["lambda_hos"] * Qhos

    dRdt = params["theta_h"] * Qh + params["theta_hos"] * Qhos - params[
        "theta"] * R
    dDdt = params["lambda_h"] * Qh + params["lambda_hos"] * Qhos
    return [
        dScdt, dSncdt, dElldt, dEhldt, dIadt, dIsdt, dIssdt, dQhdt, dQhosdt,
        dRdt, dDdt
    ]


"""
Runs the model using initial conditions and a model function and integrates it to time t
"""


def run_model(model_func, initial_conditions, time, params):
    ret = odeint(model_func, initial_conditions, time, args=(params, ))
    return ret


def predict_own_model(model_func, initial_conditions, time, params):
    predicted_data = run_model(model_func, initial_conditions, time, params)
    predicted_Sc = predicted_data[:, 0]
    predicted_Snc = predicted_data[:, 1]
    predicted_Ell = predicted_data[:, 2]
    predicted_Ehl = predicted_data[:, 3]
    predicted_Ia = predicted_data[:, 4]
    predicted_Is = predicted_data[:, 5]
    predicted_Iss = predicted_data[:, 6]
    predicted_Qh = predicted_data[:, 7]
    predicted_Qhos = predicted_data[:, 8]
    predicted_R = predicted_data[:, 9]
    predicted_D = predicted_data[:, 10]
    # return predicted_Sc, predicted_Snc, predicted_Ell, predicted_Ehl, predicted_Ia, predicted_Is, predicted_Iss, predicted_Qh, predicted_Qhos, predicted_R, predicted_D
    predicted_I = predicted_Ia + predicted_Is + predicted_Iss
    return predicted_Sc, predicted_Snc, predicted_Ell, predicted_Ehl, predicted_Ia, predicted_Is, predicted_Iss, predicted_Qh, predicted_Qhos, predicted_R, predicted_D


################################################################################
#############################Parameter Related Functions########################
################################################################################


def error(params, initial_conditions, t, gt_data, model_func):
    sol = run_model(model_func, initial_conditions, t, params)
    predicted_I = sol[:, 4] + sol[:, 5] + sol[:, 6]
    predicted_R = sol[:, 9]
    predicted_D = sol[:, 10]
    opt_sol = np.vstack((predicted_I, predicted_R, predicted_D)).T
    return (opt_sol - gt_data).ravel()


def estimate_params(confirmed_csv, recovered_csv, death_csv, population_dict,
                    country, start_date, n_simulation, n_prediction):
    I_data = get_country_data(confirmed_csv, country, start_date, n_simulation)
    R_data = get_country_data(recovered_csv, country, start_date, n_simulation)
    D_data = get_country_data(death_csv, country, start_date, n_simulation)
    #*******************************************************************************

    #***********************Initialization for states*******************************
    N0 = population_dict[country]
    Ell0 = 1
    Ehl0 = 1
    Ia0 = 1
    Is0 = 1
    Iss0 = 1
    Qh0 = 0
    Qhos0 = 0
    R0 = 0
    D0 = 0
    Sc0 = 0.5 * (N0 - Ell0 - Ehl0 - Ia0 - Is0 - Iss0 - Qh0 - Qhos0 - R0 - D0)
    Snc0 = 0.5 * (N0 - Ell0 - Ehl0 - Ia0 - Is0 - Iss0 - Qh0 - Qhos0 - R0 - D0)
    #*******************************************************************************

    #***********************Initialization for parameters***************************
    import random

    beta_s = 1
    beta_a = 1
    beta_ss = 1
    beta_Qh = 1
    beta_Qhos = 1
    alpha_c = 1
    alpha_nc = 1
    theta = 1
    theta_hos = 1
    theta_h = 1
    gamma_llc = 0.1
    gamma_llnc = 0.1
    kappa_ll = 1
    kappa_hl = 1
    rho = 0.1
    rho_hls = 0.1
    rho_hla = 0.1
    rho_lls = 0.1
    rho_lla = 0.1
    delta_ss = 1
    delta_s = 1
    delta_a = 1
    ita_ss = 0.1
    ita_s = 0.1
    lambda_hos = 1
    lambda_h = 1

    params = Parameters()
    params.add('beta_s', value=beta_s, min=0, max=10)
    params.add('beta_a', value=beta_a, min=0, max=10)
    params.add('beta_ss', value=beta_ss, min=0, max=10)
    params.add('beta_Qh', value=beta_Qh, min=0, max=10)
    params.add('beta_Qhos', value=beta_Qhos, min=0, max=10)
    params.add('alpha_c', value=alpha_c, min=0, max=10)
    params.add('alpha_nc', value=alpha_nc, min=0, max=10)
    params.add('theta', value=theta, min=0, max=10)
    params.add('theta_hos', value=theta_hos, min=0, max=10)
    params.add('theta_h', value=theta_h, min=0, max=10)
    params.add('gamma_llc', value=gamma_llc, min=0, max=1)
    params.add('gamma_llnc', value=gamma_llnc, min=0, max=1)
    params.add('kappa_ll', value=kappa_ll, min=0, max=10)
    params.add('kappa_hl', value=kappa_hl, min=0, max=10)
    params.add('rho', value=rho, min=0, max=1)
    params.add('rho_hls', value=rho_hls, min=0, max=1)
    params.add('rho_hla', value=rho_hla, min=0, max=1)
    params.add('rho_lls', value=rho_lls, min=0, max=1)
    params.add('rho_lla', value=rho_lla, min=0, max=1)
    params.add('delta_ss', value=delta_ss, min=0, max=10)
    params.add('delta_s', value=delta_s, min=0, max=10)
    params.add('delta_a', value=delta_a, min=0, max=10)
    params.add('ita_ss', value=ita_ss, min=0, max=1)
    params.add('ita_s', value=ita_s, min=0, max=1)
    params.add('lambda_hos', value=lambda_hos, min=0, max=10)
    params.add('lambda_h', value=lambda_h, min=0, max=10)
    #*******************************************************************************

    #****************************Time axis******************************************
    t = np.linspace(0, n_simulation, n_simulation)
    t_predict = np.linspace(0, n_prediction, n_prediction)
    #*******************************************************************************

    #*********************Estimating the parameters of the model*******************
    initial_conditions = [
        Sc0, Snc0, Ell0, Ehl0, Ia0, Is0, Iss0, Qh0, Qhos0, R0, D0
    ]
    gt_data = (np.vstack((I_data, R_data, D_data)).T).astype(np.float64)
    result = minimize(error,
                      params,
                      args=(initial_conditions, t, gt_data, deriv_own_model),
                      method="leastsq")

    return I_data, R_data, D_data, result.params, initial_conditions, t_predict


################################################################################
#############################Visualization functions############################
################################################################################


def visualize(country, n_days, I, R, D, I_p, R_p, D_p):
    div = Div(text="Own Model",
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
            p.circle(range(len(D[key])),
                     D[key],
                     color="brown",
                     legend_label="D")
            p.line(range(len(I[key])), I[key], color="orange", alpha=0.5)
            p.line(range(len(R[key])), R[key], color="green", alpha=0.5)
            p.line(range(len(D[key])), D[key], color="brown", alpha=0.5)

            p.square(range(len(I_p[key])),
                     I_p[key],
                     color="blue",
                     legend_label="Predicted I")
            p.square(range(len(R_p[key])),
                     R_p[key],
                     color="purple",
                     legend_label="Predicted R")
            p.square(range(len(D_p[key])),
                     D_p[key],
                     color="cyan",
                     legend_label="Predicted D")
            p.line(range(len(I_p[key])), I_p[key], color="blue", alpha=0.5)
            p.line(range(len(R_p[key])), R_p[key], color="purple", alpha=0.5)
            p.line(range(len(D_p[key])), D_p[key], color="cyan", alpha=0.5)
            p.legend.location = "top_left"
            p.legend.label_text_font_size = "7pt"
            country_layout.append(p)
        net_layout.append(country_layout)
    net_fig = gridplot(net_layout)
    net_layout = column(div, net_fig)
    return Panel(child=net_layout, title="Model Expressivity")


def visualize_600(country, n_days, initial_conditions):
    color_dict = {75: "blue", 150: "green", 225: "red", 300: "purple"}
    t_600 = np.linspace(0, 730, 730)
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
            Sc_p_600, Snc_p_600, Ell_p_600, Ehl_p_600, Ia_p_600, Is_p_600, Iss_p_600, Qh_p_600, Qhos_p_600, R_p_600, D_p_600 = predict_own_model(
                deriv_own_model, initial_conditions, t_600, params[key])
            I_p_600 = Ia_p_600 + Is_p_600 + Iss_p_600
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
            p.square(range(len(D_p_600)),
                     D_p_600,
                     color=color_dict[n],
                     legend_label="D for t=" + str(n))
            p.line(range(len(D_p_600)),
                   D_p_600,
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
    S_p, I_p, R_p = predict_own_model(deriv_own_model, initial_conditions, t,
                                      params_sweep)
    params_sweep["beta"].value = (params_sweep["beta"].value) * 2
    S_p_beta, I_p_beta, R_p_beta = predict_own_model(deriv_own_model,
                                                     initial_conditions, t,
                                                     params_sweep)

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
    S_p_gamma, I_p_gamma, R_p_gamma = predict_own_model(
        deriv_own_model, initial_conditions, t, params_sweep)

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
I = {}
R = {}
D = {}
Sc_p = {}
Snc_p = {}
Ell_p = {}
Ehl_p = {}
Ia_p = {}
Is_p = {}
Iss_p = {}
Qh_p = {}
Qhos_p = {}
R_p = {}
D_p = {}
I_p = {}
params = {}

country = ["India", "Italy", "New Zealand"]
n_days = [75, 150, 225, 300]

t = None
initial_conditions = None
for c in country:
    for n in n_days:
        key = c + "," + str(n)
        I[key], R[key], D[key], params[
            key], initial_conditions, t = estimate_params(
                confirmed_csv, recovered_csv, death_csv, population_dict, c,
                start_date, min(n, 314), n)
        Sc_p[key], Snc_p[key], Ell_p[key], Ehl_p[key], Ia_p[key], Is_p[
            key], Iss_p[key], Qh_p[key], Qhos_p[key], R_p[key], D_p[
                key] = predict_own_model(deriv_own_model, initial_conditions,
                                         t, params[key])
        I_p[key] = Ia_p[key] + Is_p[key] + Iss_p[key]
#*******************************************************************************
#****************************Saving pickle objects******************************
file_dict = {
    "I.pickle": I,
    "R.pickle": R,
    "D.pickle": D,
    "Sc_p.pickle": Sc_p,
    "Snc_p.pickle": Snc_p,
    "Ell_p.pickle": Ell_p,
    "Ehl_p.pickle": Ehl_p,
    "Ia_p.pickle": Ia_p,
    "Is_p.pickle": Is_p,
    "Iss_p.pickle": Iss_p,
    "Qh_p.pickle": Qh_p,
    "Qhos_p.pickle": Qhos_p,
    "R_p.pickle": R_p,
    "D_p.pickle": D_p,
    "I_p.pickle": I_p,
    "params.pickle": params
}
out_dir = "saved_state/"
for i in file_dict.keys():
    with open(out_dir + i, "wb") as handle:
        pickle.dump(file_dict[i], handle)
#*******************************************************************************
#***********************Visualization*******************************************
output_file("own_model_dashboard.html")

display_tabs = []
display_tabs.append(visualize(country, n_days, I, R, D, I_p, R_p, D_p))
display_tabs.append(visualize_600(country, n_days, initial_conditions))
# display_tabs.append(visualize_parameter_sweep(params, initial_conditions, t))

dashboard = Tabs(tabs=display_tabs)
show(dashboard)
#*******************************************************************************
print("END OF FILE")
################################################################################
#############################End################################################
################################################################################