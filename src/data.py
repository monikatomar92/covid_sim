#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Monika Tomar
# @Date:   2020-12-01 23:47:20
# @Last Modified by:   Monika Tomar
# @Last Modified time: 2020-12-02 00:30:25

import pandas as pd

csv_filename = "/home/monika/COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv"
csv_file = pd.read_csv(csv_filename)
csv_file.set_index("Country/Region", inplace=True)
print(csv_file.loc["India", "1/22/20":].to_numpy().shape)