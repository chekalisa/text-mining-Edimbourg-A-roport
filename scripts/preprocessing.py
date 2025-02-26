import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer 
import contractions #pip install contractions
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer 
from nltk.util import bigrams as nltk_bigrams

stop_w=stopwords.words('english')
#we add some stopwords to eliminate
stop_w.extend(['edinburgh', 'airport', 'skip', 'ok', 'na','u', 'pm'])


# Initialize lemmatizer and stemmer
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()


def cleaning_text(df, col):
    ''' this function cleans a text column (delete speciale caracters, numbers etc)
      '''
    #transform type to str
    df["suggestions_clean"]=col.astype(str)

    # Define the regular expression pattern
    pattern = r'[^\w\s]'
    col = col.apply(lambda x: re.sub(pattern, '', x)) #delete special characters
    number_pattern = r'\d+' 
    col = col.apply(lambda x: re.sub(number_pattern, '', x)) #delete numbers
    col = col.apply(lambda x: x.lower()) #lowercase
    col = col.apply(lambda x: contractions.fix(x)) #dealing with contraction
    return col




def preprocess_stemming(text):

    ''' this function deletes stopwords, tokenizes text and does stemming
    it returns stemmed column
    '''
    #remove stopwords
    
    tokens = word_tokenize(text)
    filtered_words = [word for word in tokens if word not in stop_w] 
    
    #Stemming
    
    stemmed_tokens = [stemmer.stem(token) for token in filtered_words]

    return stemmed_tokens

def preprocess_lemming(text):

    '''this function deletes stopwords, tokenizes text and does lemmatization
    it returns lemmatized column'''

    # Tokenize text
    tokens = word_tokenize(text)
    
    # Remove stopwords
    filtered_words = [word for word in tokens if word not in stop_w]
    
    
    lemm_tokens = [lemmatizer.lemmatize(token) for token in filtered_words]
    lemm_text= ' '.join(lemm_tokens)
    
    return lemm_text

def bigrams(text, exclude_bigrams=[]):

    ''' this function returns bigrams'''

    # Ensure input is a string
    if isinstance(text, list):
        text = ' '.join(text)
    
    # Tokenize text
    tokens = word_tokenize(text)
    
    # Remove stopwords
    filtered_words = [word for word in tokens if word not in stop_w]
    
    # Create bigrams
    bigram_tokens = ["_".join(bigram) for bigram in nltk_bigrams(filtered_words)]
    
    # Exclude specific bigrams
    bigram_tokens = [bigram for bigram in bigram_tokens if bigram not in exclude_bigrams]
    
    return bigram_tokens


def tokenize(text):

    ''' simple tokenization of the text'''
    
    tokens = word_tokenize(text)  # Tokenize the text
    tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Remove stopwords and lemmatize
    return tokens

def cleaning_text_sug(df, col):
    "Function to clean text for Sentiment Analysis"
    #transform type to str
    df["suggestions_clean"]=col.astype(str)
    # Define the regular expression pattern
    pattern = r'[^\w\s]'
    col = col.apply(lambda x: re.sub(pattern, '', x)) #delete special characters
    number_pattern = r'\d+' 
    col = col.apply(lambda x: re.sub(number_pattern, '', x)) #delete numbers
    col = col.apply(lambda x: x.lower()) #lowercase
    col = col.apply(lambda x: contractions.fix(x)) #dealing with contraction
    return col

def cleaning_text_ser(df, col):
    "Function to clean text for Sentiment Analysis"
    #transform type to str
    df["services_clean"]=col.astype(str)
    # Define the regular expression pattern
    pattern = r'[^\w\s]'
    col = col.apply(lambda x: re.sub(pattern, '', x)) #delete special characters
    number_pattern = r'\d+' 
    col = col.apply(lambda x: re.sub(number_pattern, '', x)) #delete numbers
    col = col.apply(lambda x: x.lower()) #lowercase
    col = col.apply(lambda x: contractions.fix(x)) #dealing with contraction
    return col

 
