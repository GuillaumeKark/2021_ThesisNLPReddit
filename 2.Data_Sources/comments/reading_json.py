# -*- coding: utf-8 -*-
"""
Date: April 2021
Comment: This program reads the json file in chunks and select the range 1Jan2020 to end of dataset.
The objective is to immediately remove useless rows in the analysis to improve speed and allow 
memory storage of the dataset in pandas without chunkload.
"""

import pandas as pd
PATH = "" 

TIMESTAMP = 1577836800 #1Jan2020-00:00:00
CHUNKSIZE = 100000 #large chunk size to speed up the file
#24,840,282 rows
chunks = pd.read_json(PATH +'final_wsb_archived.jsonl',
                        orient="records", lines=True, chunksize=CHUNKSIZE)


data_filtered = pd.DataFrame()
i = 0

#Run and filter by date to shrink size
for c in chunks:
    i = i +1
    c = c[c["created_utc"] >= TIMESTAMP] #post after 1Jan2020
    print(c, str(CHUNKSIZE* i))
    if data_filtered.empty:
        data_filtered = c
    else:
        data_filtered = pd.concat([data_filtered, c], ignore_index=True)

#Save as csv
#dimensions: (18m x 8)
data_filtered.to_csv(PATH + 'filtered_bydate.csv', index=False)
#data is 2x smaller in the final CSV than the inital JSON.
