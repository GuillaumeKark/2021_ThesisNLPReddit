# -*- coding: utf-8 -*-
"""
April 2021
Comment: This code queries all submissions on a specific date and save them in the subdirectory under the format timestamp.csv.
If comments are missing the file is saved as empty and returns a warning message.
"""

import pandas as pd
from pmaw import PushshiftAPI
api = PushshiftAPI(num_workers=60) #12 cores on computer. See PMAW Documentation for the right number of num_workers. https://pypi.org/project/pmaw/
import datetime as dt
import time


SUB_DIR = "" #put your subdirectory HERE. The subdir will contains NUM_DAYS files.
NUM_DAYS = 1 #set the number of days to query. For the range covered the number was 465.

start_time = int(dt.datetime(2020, 1, 1, tzinfo=dt.timezone.utc).timestamp())

for i in range(0,9):
    start = time.time()
    subreddit = "wallstreetbets"
    limit = 500000 #can't set a dynamic limit with PMAW. With statistics from wsb, we know that 500k submssions are never reached.
    submissions = api.search_submissions(subreddit=subreddit, limit=limit, before=start_time+86400, after=start_time)
    print(f'Retrieved {len(submissions)} comments from Pushshift')
    submissions_df = pd.DataFrame(submissions)
    print("--- %s seconds ---" % (time.time() - start))
    
    #if dataset contains submissions, else except.
    try:
        submissions_df = submissions_df[['title','created_utc','author','score','link_flair_text','permalink','id', 'pinned']]
        submissions_df.to_csv(SUB_DIR + str(start_time)+".csv", index=False)
    except:
        submissions_df.to_csv(SUB_DIR + str(start_time)+".csv", index=False)
    print(dt.datetime.fromtimestamp(start_time))
    start_time = start_time + 86400 #move to next day (+86,400 seconds)
