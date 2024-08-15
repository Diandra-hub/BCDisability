import pandas as pd
import re
import nltk
from textblob import TextBlob
from gensim import corpora
import gensim
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
# Load the cleaned data
df = pd.read_csv('cleaned_bcdisability_posts.csv')

# Remove duplicates and handle missing values
df = df.drop_duplicates()
df = df.dropna()

# Text preprocessing
def preprocess_text(text):
    if isinstance(text, str):
        # Convert to lowercase
        text = text.lower()
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = word_tokenize(text)
        filtered_tokens = [word for word in tokens if word not in stop_words]
        return ' '.join(filtered_tokens)
    return ''

# Apply text preprocessing
df['processed_body'] = df['body'].apply(preprocess_text)

# Sentiment Analysis
df['sentiment'] = df['processed_body'].apply(lambda x: TextBlob(x).sentiment.polarity)

# Topic Modeling
processed_docs = df['processed_body'].apply(lambda x: x.split())
dictionary = corpora.Dictionary(processed_docs)
corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

# Build LDA model
lda_model = gensim.models.LdaModel(corpus, num_topics=5, id2word=dictionary, passes=15)

# Display topics
topics = lda_model.print_topics(num_words=5)
for topic in topics:
    print(topic)