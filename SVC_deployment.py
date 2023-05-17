from joblib import load
#Import necessary packages
import os
import numpy as np
import pandas as pd
import sklearn
import nltk
import spacy

#Importing tools from nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
#Initializing the SpaCy NLP pipeline with the necessary components for lemmatization
nlp = spacy.load('en_core_web_sm')

# Download the required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
# Creating the set of stop words and initializing the snowball stemmer
spacy_stop_words = set(spacy.lang.en.stop_words.STOP_WORDS)
nltk_stop_words = nltk_stop_words = set(stopwords.words('english'))
stop_words = nltk_stop_words.union(spacy_stop_words)
stemmer = SnowballStemmer(language = 'english')

# Defining a function to applying stemming to the text using the NLTK SnowballStemmer
def stem_text(text):
    # Tokenize the text using the NLTK word_tokenize function
    text = word_tokenize(text)
    # Remove any tokens with non-alphabetical characters and lowercase
    text = [word.lower() for word in text if word.isalpha()]
    # Stem non-stop words using the SnowballStemmer from NLTK
    text = [stemmer.stem(word) for word in text if word not in stop_words]
    # Return the tokenized text
    return text

# This dictionary maps the numerical labels to the string labels
label_string = {
   0: 'neutral',
   1: 'confusion/curiosity',
   2: 'realization/surprise',
   3: 'sadness/grief',
   4: 'approval/pride',
   5: 'admiration',
   6: 'fear/nervousness/embarrassment',
   7: 'gratitude/relief',
   8: 'joy/amusement',
   9: 'remorse/disappointment',
   10: 'desire/excitement/optimism',
   11: 'annoyance/anger',
   12: 'disapproval/disgust',
   13: 'love/caring'
}

# Loading the trained support vector classifier
model = load('./savedmodel/support_vector_classifier.joblib')


while(True):
    user_text_input = [str(input("Enter Text: "))]
    text_prediction = model.predict(user_text_input)
    print(label_string[text_prediction[0]])