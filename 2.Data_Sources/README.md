Datasets are too heavy to publish on the GitHub.
Submissions dataset is 200Mo and Comments dataset is 7.5Go unfiltered.
However, if you need the data, do not hesitate to send me an email.

/submissions contains the code used to query and merge the submissions.
/comments contains the code used to reduce the size of the the comment dataset and fit in memory using pandas without batch loading.
/gme_stock_price contains the financial data of GME on the relevant dates.

submissions cover the range of date from 1Jan2020 to 2Apr2021. There are few missing days and some imcomplete. This is a known Pushift issue. See Master Thesis for details.
For example, timestamps 1604534400, 1612656000 are known incomplete.
1612483200, 1612569600, 1614556800, 1614988800 and the range 1616025600 - 1616716800 are missing.


comments cover all comment since the creation of the subreddit until 31Jan2021. However after 15Jan2021, comments are incomplete and the construction remains a black box.
*"The project of archiving WSB, as I see it, can be broken up into the following segments of data (done):
Comments until December 2019, available in the pushshift.io archives.
Comments from December 2019-when I started recording new comments (~2 weeks ago), which had to be archived from the pushshift API, slowly, over a period of roughly a week.
Comments from since I’ve begun the project, which I’ve been streaming live onto my server from PRAW."*
See https://www.reddit.com/r/DataHoarder/comments/l7oxw9/creating_a_wallstreetbets_archive/ for more details.
