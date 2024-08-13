# Reddit analysis code (using data extraction in file: bcdisability_posts.csv)
import pandas as pd

# Load the data (df = data file)
df = pd.read_csv('cleaned_bcdisability_posts.csv') 

# Clean data:
# 1) Remove duplicates
df = df.drop_duplicates()

# 2) Handle missing values
df = df.dropna()

# Sentiment Analysis
from textblob import TextBlob
df['sentiment'] = df['body'].apply(lambda x: TextBlob(x).sentiment.polarity)

# Topic Modeling
import gensim
from gensim import corpora

processed_docs = df['body'].apply(lambda x: x.lower().split())
dictionary = corpora.Dictionary(processed_docs)
corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
lda_model = gensim.models.LdaModel(corpus, num_topics=5, id2word=dictionary, passes=15)
topics = lda_model.print_topics(num_words=5)
for topic in topics:
    print(topic)

# Output: (topic_number, 'weight*"word" + weight*"word" + ...') 
# (Using the topic modeling process using Latent Dirichlet Allocation (LDA).)
# Topic Number: The number at the start (e.g., 0, 1, 2) is the index of the topic.
# Words in the Topic: Each topic is represented by a list of words that are most associated with that topic.
# Weights: The numbers before each word (e.g., 0.043, 0.031) are the weights or probabilities,
# indicating the importance of each word in defining the topic.
# A higher weight means the word is more strongly associated with that topic.