# -*- coding: utf-8 -*-
"""
UNUSED IN THE FINAL VERSION.
April 2021
Comment: This code uses the PRAW API to collect posts using the Reddit API. However because of time limitations it was unused. (slow query time and max 100 posts).
I also tried to used a code based on PRAW tha requeries back on the last timestamp. It works well, however it is a very slow process so Pushift is the way to go.
Nevertheless, the Reddit API is the only way to build a live monitoring system, so here is the code.
"""
import praw
import pandas as pd

posts = []

#put IDs from the API in the praw. See https://github.com/reddit-archive/reddit/wiki/OAuth2 
reddit = praw.Reddit(client_id='', client_secret='', user_agent='')

start_time = 1604534400
stop_time = 1604620800
ml_subreddit = reddit.subreddit('MachineLearning') #put the subreddit name here

for post in ml_subreddit.new(limit=100):
    posts.append([post.title, post.score, post.id, post.subreddit, post.url, post.num_comments, post.selftext, post.created])
posts = pd.DataFrame(posts,columns=['title', 'score', 'id', 'subreddit', 'url', 'num_comments', 'body', 'created'])
print(posts)