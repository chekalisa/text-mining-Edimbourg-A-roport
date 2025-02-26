import pandas as pd
import numpy as np

import spacy
nlp = spacy.load('en_core_web_sm')

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

from scipy.stats import pearsonr

def sentiment_score(df,col1, col2):
    """
    This function is for calculate the polarity score of each opinion, allowing to see if that opinion is positive, negative or neutral

    Input:
    df: dataframe used
    col1, col2: 2 text columns

    Output: 2 new columns for polarity scores
    df['sentiment_suggestions']: polarity score of each general suggestion
    df['sentiment_services']: polarity score of each suggestion on premium services
    """

    df['sentiment_suggestions'] = col1.apply(lambda x: analyzer.polarity_scores(x)['compound'])
    df['sentiment_services'] = col2.apply(lambda x: analyzer.polarity_scores(x)['compound'])

def aggregate_sentiment(df,col1, col2):
    """
    This function is to calculate the average polarity score of 2 text columns
    
    Input:
    df: dataframe used
    col1, col2: 2 text columns

    Output: a new column of average polarity score of each client
    """
    sentiment_score(df,col1, col2)
    df['combined_sentiment'] = (df['sentiment_suggestions'] + df['sentiment_services']) / 2.0

# Calculate average sentiment for each aspect
def avg_sentiment_aspect(feedbacks, aspects):
    """
    This function is to calculate the average polarity score for each aspect

    Input:
    feedbacks: text column
    aspects: list of elements we want to define for a specific aspect

    Output: average polarity score for each aspect 
    """
    aspect_sentiments = {aspect: 0 for aspect in aspects}  # Initialize sentiment scores for each aspect
    # Iterate over each sentence in the list
    for sentence in feedbacks:
        # Calculate the sentiment score of the current sentence
        sentiment = analyzer.polarity_scores(sentence)['compound']
        # Check if each aspect is mentioned in the current sentence
        for aspect in aspects:
            if aspect in sentence:
                # Add the sentiment score to the corresponding aspect's total score
                aspect_sentiments[aspect] += sentiment
    # Calculate average sentiment for each aspect
    for aspect, sentiment in aspect_sentiments.items():
        aspect_sentiments[aspect] /= len(feedbacks)
    return aspect_sentiments
    