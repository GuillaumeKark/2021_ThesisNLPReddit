# -*- coding: utf-8 -*-
"""
April 2021
Comment: This code merges all csv filed recovered with submissions_query.py.
"""

import numpy as np
import pandas as pd

SUB_DIR = "" #directory of queried sub-files. Same as in submissions_query.
SAVE_DIR = "" #directory to save the final dataset.

timestamp = 1577836800 #firt record
last_timestamp = 1617926400 #last record

export_ds = pd.DataFrame()

while timestamp != last_timestamp:
    
    try:
        import_ds = pd.read_csv(SUB_DIR + str(timestamp) + ".csv")
        import_ds = import_ds[['title','created_utc','author','score','link_flair_text','permalink','id', 'pinned']]
        export_ds = pd.concat([export_ds, import_ds], axis=0) 
        print("Timestamp " + str(timestamp) + " merged.")
        
    except:
        print("Timestamp "  + str(timestamp) + " is empty.") #After query, print if the day is empty.

    
    timestamp = timestamp + 86400 #move to next day (=file).

#drop duplicates 
export_ds = export_ds.drop_duplicates(subset ="title", keep = "first")

#sort by ascending date and reset indexes
export_ds = export_ds.sort_values(by="created_utc", ignore_index=True)

#create a variable for the day when submission has been published.
export_ds["real_datetime"] = pd.to_datetime(export_ds["created_utc"], unit='s').dt.floor('d')

#is GME contained in the submissions, we check the to most comment occurences. Game Stop and Game-Stop are almost never used on WSB.
export_ds["contains_gme"] = export_ds["title"].str.contains("gme|gamestop", case=False, na=False)


export_ds.to_csv(SAVE_DIR + "submissions.csv", index=False)
