
import praw
import pandas as pd

# Fill in your Reddit app details
client_id = 'dYSoaMBxeRsUqidUeIo53Q'
client_secret = 'aNYoOSslntv_rycZDrorHXjwcTBjlQ'
user_agent = 'B.C.Disability'

# Connect to Reddit
reddit = praw.Reddit(client_id=client_id, client_secret=client_secret, user_agent=user_agent)

# Select the subreddit
subreddit = reddit.subreddit('disability')

# Extract posts
posts = []
for post in subreddit.hot(limit=100):
    posts.append([post.title, post.selftext, post.score, post.id, post.subreddit, post.url, post.num_comments, post.created])

# Create a DataFrame
posts_df = pd.DataFrame(posts, columns=['title', 'body', 'score', 'id', 'subreddit', 'url', 'num_comments', 'created'])

# Save to a CSV file
posts_df.to_csv('bcdisability_posts.csv', index=False)