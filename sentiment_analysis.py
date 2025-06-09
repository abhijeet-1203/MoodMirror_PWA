from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
import plotly.express as px  # Add this with other imports
import random  # Add this with other imports

nltk.download('punkt')

analyzer = SentimentIntensityAnalyzer()

def analyze_sentiment_vader(text):
    score = analyzer.polarity_scores(text)
    sentiment = 'Neutral'
    if score['compound'] >= 0.05:
        sentiment = 'Positive'
    elif score['compound'] <= -0.05:
        sentiment = 'Negative'
    return sentiment, score['compound']

def analyze_sentiment_textblob(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    sentiment = 'Neutral'
    if polarity > 0:
        sentiment = 'Positive'
    elif polarity < 0:
        sentiment = 'Negative'
    return sentiment, polarity

def get_keywords(text):
    blob = TextBlob(text)
    return blob.noun_phrases
