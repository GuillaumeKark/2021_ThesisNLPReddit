# -*- coding: utf-8 -*-
"""
April 2021
Comment: This code queries all months of data (by minute) for GME and concanates it.
"""

import pandas as pd

TEMP_PATH = ""
OUTPUT_PATH = ""


#query system, use your own API KEY
for i in range(1,13):
    link = "https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY_EXTENDED&symbol=GME&interval=1min&slice=year1month"+ str(i) + "&apikey="
    dataframe = pd.read_csv(link)
    dataframe.to_csv(TEMP_PATH + str(i) + ".csv")
    print(i)


output = pd.DataFrame()
#merging system
for i in range(1,13):
    read = pd.read_csv(TEMP_PATH + str(i) + ".csv")
    output = pd.concat([output, read], axis=0)
     
output.to_csv(OUTPUT_PATH + "stock_bymin.csv")
