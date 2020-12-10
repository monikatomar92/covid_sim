# -*- coding: utf-8 -*-
# @Author: Monika Tomar
# @Date:   2020-12-06 21:57:41
# @Last Modified by:   Monika Tomar
# @Last Modified time: 2020-12-07 00:02:59

from collections import defaultdict
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
import json

################################################################################
#############################Read pickle objects into dict######################
################################################################################

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

country = ["India", "Italy", "New Zealand"]
n_days = [75, 150, 225, 300]

stats_list = {
    "Sc_p.pickle", "Snc_p.pickle", "Ell_p.pickle", "Ehl_p.pickle",
    "Ia_p.pickle", "Is_p.pickle", "Iss_p.pickle", "Qh_p.pickle",
    "Qhos_p.pickle", "R_p.pickle", "D_p.pickle", "I_p.pickle"
}
operations_dict = {"max": np.max, "sum": np.sum, "std_dev": np.std}
summary_stats_dict = defaultdict(
    lambda: defaultdict(lambda: defaultdict(float)))
country_list = ["India", "Italy", "New Zealand"]

out_dir = "saved_state/"
for i in file_dict.keys():
    with open(out_dir + i, "rb") as handle:
        file_dict[i] = pickle.load(handle)

for j in stats_list:
    for k in operations_dict:
        for l in country_list:
            key = l + "," + str(300)
            summary_stats_dict[j][l][k] = operations_dict[k](file_dict[j][key])

with open("summary_stats.json", "w") as json_file:
    json.dump(dict(summary_stats_dict),
              json_file,
              sort_keys=True,
              indent=4,
              default=lambda o: o._asdict())

params_dict = defaultdict(Parameters)
for l in country_list:
    key = l + "," + str(300)
    params_dict[key] = file_dict["params.pickle"][key]
print(params_dict)

with open("params.txt", "w") as json_file:
    json_file.write(str(dict(params_dict)))

################################################################################
#############################Visualizations#####################################
################################################################################
