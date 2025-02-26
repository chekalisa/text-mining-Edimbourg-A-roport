from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
import random
import numpy as np
import pandas as pd


# Count of number of words (construct term-documentary matrix)
count = CountVectorizer(
    analyzer='word',
    # minimum frequency of a term to be included in the term-document matrix
    min_df=10,
    # bigrams, trigrams
    ngram_range=(2,3))

#LDA model
lda_model = LatentDirichletAllocation(
     n_components=6,
    learning_method='online',
    random_state=0,
    n_jobs=-1,
)


def create_vocabulary(docs):
    '''
    This function creates the vocabulary from a list of documents

    Arguments:
    docs : a list of documents with each document being a list of words

    Returns:
    vocab_size : the size of the vocabulary
    word2id : dictionary mapping words to their respective IDs
    '''
    all_bigrams = list(set(word for doc in docs for word in doc))
    vocab_size = len(all_bigrams)
    word2id = {word: i for i, word in enumerate(all_bigrams)}
    return vocab_size, word2id

def initialize_matrices(docs, num_topics, vocab_size, word2id, alpha=0.1, beta=0.1):
    '''
This function initializes matrices and counts required for topic modeling using Gibbs Sampling. 
It initializes the topic assignments for words in documents, counts the number of words assigned to each topic in each document,
counts the number of times each word is assigned to each topic, and calculates the total count of words assigned to each topic. 
The function returns these initialized matrices and counts.

Arguments:
docs: a list of documents where each document is represented as a list of words
num_topics: an integer indicating the number of topics
vocab_size: an integer indicating the size of the vocabulary, derived from the previous function in this case
word2id: dictionary mapping words to their respective IDs, derived from the previous function
alpha: a hyperparameter that controls the document-topic distribution, float, optional, it controls the distribution of topics over documents
beta: a hyperparameter that controls the document-topic distribution, float, optional, it controls the distribution of words over topics

Returns:
topic_assignments: list of lists that represents the initial assignments of topics to words in documents
doc_topic_counts: 2D NumPy array that represents the count of words assigned to each topic in each document
topic_word_counts: 2D NumPy array that represents the count of each word assigned to each topic
topic_counts: 1D NumPy array that represents the total count of words assigned to each topic


'''
    doc_count = len(docs)
    topic_assignments = []
    for doc in docs:
        current_doc_topics = [random.randint(0, num_topics - 1) for _ in doc]
        topic_assignments.append(current_doc_topics)

    doc_topic_counts = np.zeros((doc_count, num_topics)) + alpha
    topic_word_counts = np.zeros((num_topics, vocab_size)) + beta
    topic_counts = np.zeros(num_topics) + vocab_size * beta

    for d_idx, doc in enumerate(docs):
        for w_idx, word in enumerate(doc):
            topic = topic_assignments[d_idx][w_idx]
            word_id = word2id[word]
            doc_topic_counts[d_idx][topic] += 1
            topic_word_counts[topic][word_id] += 1
            topic_counts[topic] += 1

    return topic_assignments, doc_topic_counts, topic_word_counts, topic_counts

def sample_new_topic(d, word_id, current_topic, doc_topic_counts, topic_word_counts, topic_counts, num_topics):
    '''
    This function updates the topic assignment for a specific word in a document using Gibbs Sampling

    Arguments:
    d: the index of the document
    word_id: the ID of the word
    current_topic: current topic assignment for the word
    doc_topic_counts: 2D NumPy array representing the count of words assigned to each topic in each document
    topic_word_counts: 2D NumPy array representing the count of each word assigned to each topic
    topic_counts: 1D NumPy array representing the total count of words assigned to each topic
    num_topics: total number of topics

    Returns:
    new_topic: new topic assignment for the word
    '''

    doc_topic_counts[d][current_topic] -= 1
    topic_word_counts[current_topic][word_id] -= 1
    topic_counts[current_topic] -= 1

    topic_probs = (doc_topic_counts[d] * topic_word_counts[:, word_id]) / topic_counts
    topic_probs /= np.sum(topic_probs)

    new_topic = np.random.choice(np.arange(num_topics), p=topic_probs)

    doc_topic_counts[d][new_topic] += 1
    topic_word_counts[new_topic][word_id] += 1
    topic_counts[new_topic] += 1

    return new_topic

def gibbs_sampling(docs, topic_assignments, doc_topic_counts, topic_word_counts, topic_counts, word2id, num_topics, num_iterations=20):
    '''
    This function performs Gibbs Sampling to update topic assignments for words in documents.
    It iterates over each word in each document for the specified number of iterations. 
    For each word, it calculates the conditional distribution of topics given the document and word, 
    samples a new topic assignment using this distribution, 
    and updates the topic assignment matrix accordingly.

    Arguments:
    docs: a list of documents, where each document is represented as a list of words
    topic_assignments: a list of lists representing the current assignments of topics to words in documents
    doc_topic_counts: 2D NumPy array representing the count of words assigned to each topic in each document
    topic_word_counts: 2D NumPy array representing the count of each word assigned to each topic
    topic_counts: 1D NumPy array representing the total count of words assigned to each topic
    word2id: dictionary mapping words to their respective IDs
    num_topics: total number of topics
    num_iterations: number of iterations for Gibbs Sampling 

    '''
    for it in range(num_iterations):
        for d_idx, doc in enumerate(docs):
            for w_idx, word in enumerate(doc):
                current_topic = topic_assignments[d_idx][w_idx]
                word_id = word2id[word]
                new_topic = sample_new_topic(d_idx, word_id, current_topic, doc_topic_counts, topic_word_counts, topic_counts, num_topics)
                topic_assignments[d_idx][w_idx] = new_topic

def get_top_bigrams(topic_word_counts, word2id, num_bigrams=10):
    '''
    This function extracts the top bigrams for each topic based on their word counts in the topic-word matrix

    Arguments:
    topic_word_counts: 2D NumPy array representing the count of each word assigned to each topic
    word2id: dictionary mapping words to their respective IDs
    num_bigrams: number of top bigrams to extract for each topic 

    Returns:
    a dictionary where keys are topic indices and values are lists of top bigrams for each topic
    Each list contains the specified number of top bigrams
    '''

    id2word = {i: word for word, i in word2id.items()}
    top_bigrams = {}

    for topic_idx, word_counts in enumerate(topic_word_counts):
        top_word_ids = np.argsort(word_counts)[-num_bigrams:][::-1]
        top_words = [id2word[word_id] for word_id in top_word_ids]

        bigrams = [word for word in top_words if "_" in word]
        if len(bigrams) < num_bigrams:
            bigrams = [word for word in top_words if "_" in word][:num_bigrams]

        top_bigrams[topic_idx] = bigrams

    return top_bigrams



def plot_top_bigrams_by_topic(topic_word_counts, word2id, num_bigrams=10):
    '''
    This function plots the top bigrams by topic based on their frequencies

    Arguments:
    topic_word_counts: 2D NumPy array representing the count of each word assigned to each topic
    word2id: dictionary mapping words to their respective IDs
    num_bigrams: number of top bigrams to plot for each topic 

    Returns:
    A bar plot showing the top bigrams for each topic based on their frequencies
    Each subplot represents a topic, and the x-axis represents the frequency of each bigram
    '''

    bigram_data = []

    for topic_idx, word_counts in enumerate(topic_word_counts):
        bigram_freq = {word: word_counts[word2id[word]] for word in word2id if "_" in word and word_counts[word2id[word]] > 1}  
        for bigram, freq in bigram_freq.items():
            bigram_data.append({"Topic": topic_idx, "Bigram": bigram, "Frequency": freq})

    bigram_df = pd.DataFrame(bigram_data)

    topics = bigram_df["Topic"].unique()
    
    fig, axes = plt.subplots(nrows=len(topics), ncols=1, figsize=(10, 5 * len(topics)))
    if len(topics) == 1:
        axes = [axes]

    for idx, topic in enumerate(topics):
        topic_data = bigram_df[bigram_df["Topic"] == topic]
        top_bigrams = topic_data.nlargest(num_bigrams, "Frequency")
        
        axes[idx].barh(top_bigrams["Bigram"], top_bigrams["Frequency"], color='skyblue')
        axes[idx].set_title(f"Top {num_bigrams} Bigrams for Topic {topic}")
        axes[idx].invert_yaxis()  # Highest bars at the top

    plt.tight_layout()
    plt.show()


