import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import itertools
import nltk
from nltk.util import ngrams
nltk.download('vader_lexicon')
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer()
def analyze_frequency(df, remove={}, subject=" "):
    '''
    Arguments:
    df : DataFrame to use
    remove : words that we don't need to use during the frequency count 
    subject : to specify the concrete subject that we analyze

    Returns:
    The function returns a graph with the most popular words and bigrams used in the df
    '''
    # Combine all lists of words into a single list
    all_tokens = list(itertools.chain(*df['tokenized_text']))

    # Set of strings to remove
    strings_to_remove = remove

    # List comprehension to remove specific strings
    all_tokens = [item for item in all_tokens if item not in strings_to_remove]

    # Unigrams
    unigram_counts = Counter(all_tokens)
    most_common_unigrams = unigram_counts.most_common(15)
    unigram_labels = [word for word, count in most_common_unigrams]
    unigram_counts = [count for word, count in most_common_unigrams]

    # Bigrams
    bigram_tokens = list(ngrams(all_tokens, 2))
    bigram_counts = Counter(bigram_tokens)
    most_common_bigrams = bigram_counts.most_common(15)
    bigram_labels = [' '.join(bigram) for bigram, count in most_common_bigrams]
    bigram_counts = [count for bigram, count in most_common_bigrams]

    # Create the subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 8))

    # Plot Unigrams
    ax1.barh(unigram_labels, unigram_counts, color='skyblue')
    ax1.set_xlabel('Frequency')
    ax1.set_ylabel('Words')
    ax1.set_title(f'Histogram of the most popular words with {subject}')
    ax1.invert_yaxis()  # Invert the y-axis to have the most frequent words on top

    # Plot Bigrams
    ax2.barh(bigram_labels, bigram_counts, color='skyblue')
    ax2.set_xlabel('Frequency')
    ax2.set_ylabel('Bigrams')
    ax2.set_title(f'Histogram of the most popular bigrams with {subject}')
    ax2.invert_yaxis()  # Invert the y-axis to have the most frequent bigrams on top

    # Adjust layout
    plt.tight_layout()
    plt.show()

def frequency(data, remove={}, subject=" "):
    '''
    Arguments:
    df : DataFrame to use
    remove : words that we don't need to use during the frequency count 
    subject : to specify the concrete subject that we analyze

    Returns:
    The function returns a graph with the most popular words used in the df
    '''
    # Count the occurrences of words in reviews
    all_tokens = list(itertools.chain(*data['tokenized_text']))

    # Set of strings to remove
    strings_to_remove = remove

    # List comprehension to remove specific strings
    all_tokens = [item for item in all_tokens if item not in strings_to_remove]

    # Count the occurrences of each word
    word_counts = Counter(all_tokens)

    # Select the 20 most frequent words
    most_common_words = word_counts.most_common(10)

    # Extract words and their frequencies for display
    words = [word for word, count in most_common_words]
    counts = [count for word, count in most_common_words]

    # Create the histogram
    plt.figure(figsize=(10, 8))
    plt.barh(words, counts, color='skyblue')
    plt.xlabel('Frequency')
    plt.ylabel('Words')
    plt.title(f'Histogram of the most popular words with {subject}')
    plt.gca().invert_yaxis()  # Invert the y-axis to have the most frequent words on top
    plt.show()

def bigrams(df, remove={}, subject=" "):
    '''
    Arguments:
    df : DataFrame to use
    remove : words that we don't need to use during the frequency count 
    subject : to specify the concrete subject that we analyze

    Returns:
    The function returns a graph with the most popular bigrams used in the df
    ''' 
    # Combine all lists of words into a single list
    all_tokens = list(itertools.chain(*df['tokenized_text']))

    # Set of strings to remove
    strings_to_remove = remove

    # List comprehension to remove specific strings
    all_tokens = [item for item in all_tokens if item not in strings_to_remove]

    # Create bigrams using tokens
    bigrams = list(ngrams(all_tokens, 2))

    # Count occurrences of each bigram
    bigram_counts = Counter(bigrams)

    # Select the 20 most frequent bigrams
    most_common_bigrams = bigram_counts.most_common(10)

    # Extract bigrams and their frequencies for display
    bigram_labels = [' '.join(bigram) for bigram, count in most_common_bigrams]
    counts = [count for bigram, count in most_common_bigrams]

    # Create the histogram
    plt.figure(figsize=(12, 8))
    plt.barh(bigram_labels, counts, color='skyblue')
    plt.xlabel('Frequency')
    plt.ylabel('Bigrams')
    plt.title(f'Histogram of the most popular bigrams with {subject}')
    plt.gca().invert_yaxis()  # Invert the y-axis to have the most frequent bigrams on top
    plt.show()

def sentiment(df, sujet):
    '''
    Arguments:
    df : DataFrame to use
    
    Returns:
    The function returns a histogram showing the proportion of negative and positive feedback in the df
    '''
    # Categorize comments as positive or negative
    positive_comments = 0
    negative_comments = 0

    for text in df["suggestions_clean"]:
        score = sia.polarity_scores(text)
        if score['compound'] >= 0.05:
            positive_comments += 1
        elif score['compound'] <= -0.05:
            negative_comments += 1

    # Frequencies of positive and negative comments
    frequencies = {'Positive': positive_comments, 'Negative': negative_comments}

    # Create a histogram of comment frequencies
    fig, ax = plt.subplots(figsize=(8, 6))

    categories = list(frequencies.keys())
    counts = list(frequencies.values())

    ax.bar(categories, counts, color=['green', 'red'])

    # Add labels and title
    ax.set_xlabel('Sentiment')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Frequency of Positive and Negative Comments about {sujet}')

    # Display the histogram
    plt.tight_layout()
    plt.show()

def contains(df,col="feedbacks", word=None):
   
   ''' Args : 

   df : Data Frame
   col : specific column, by default column "feedbacks"
   word : word to find, by default "None"

   Returns: data frame that contains the specified word

   '''
   df_contain=df[df[col].str.contains(word, case=False, na=False)]
   return df_contain

def tables_freq(df, remove={}, subject=" "):
    
    '''
    Arguments:
    df : DataFrame to use
    remove : words that we don't need to use during the frequency count 
    subject : to specify the concrete subject that we analyze

    Returns:
    The function prints tables with the most popular words and bigrams used in the df
    '''
    # Combine all lists of words into a single list
    all_tokens = list(itertools.chain(*df['tokenized_text']))

    # Set of strings to remove
    strings_to_remove = remove

    # List comprehension to remove specific strings
    all_tokens = [item for item in all_tokens if item not in strings_to_remove]

    # Unigrams
    unigram_counts = Counter(all_tokens)
    most_common_unigrams = unigram_counts.most_common(15)
    unigram_labels = [word for word, count in most_common_unigrams]
    unigram_counts = [count for word, count in most_common_unigrams]

    # Bigrams
    bigram_tokens = list(ngrams(all_tokens, 2))
    bigram_counts = Counter(bigram_tokens)
    most_common_bigrams = bigram_counts.most_common(15)
    bigram_labels = [' '.join(bigram) for bigram, count in most_common_bigrams]
    bigram_counts = [count for bigram, count in most_common_bigrams]

    # Create DataFrames for unigrams and bigrams
    unigram_df = pd.DataFrame({
        'Word': unigram_labels,
        'Frequency': unigram_counts
    })

    bigram_df = pd.DataFrame({
        'Bigram': bigram_labels,
        'Frequency': bigram_counts
    })

    # Concatenate the DataFrames side by side
    combined_df = pd.concat([unigram_df, bigram_df], axis=1)
    
    # Print the combined DataFrame
    print(f"Table of the most popular words and bigrams with {subject}:\n")
    print(combined_df.to_string(index=False))