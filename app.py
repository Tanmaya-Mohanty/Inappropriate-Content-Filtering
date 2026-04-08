import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer
import os
import gdown

nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

vectorizer_id = st.secrets["VECTORIZER_ID"]
model_id = st.secrets["MODEL_ID"]

vectorizer_url = f"https://drive.google.com/uc?id={vectorizer_id}"
spam_classifier_url = f"https://drive.google.com/uc?id={model_id}"

# Download vectorizer and model
if not os.path.exists("content_vectorizer.pkl"):
    gdown.download(vectorizer_url, "content_vectorizer.pkl", quiet=False)

if not os.path.exists("content_classifier.pkl"):
    gdown.download(spam_classifier_url, "content_classifier.pkl", quiet=False)

# Load vectorizer and model
tfidf = pickle.load(open('content_vectorizer.pkl', 'rb'))
mnb = pickle.load(open('content_classifier.pkl', 'rb'))

def transform_text(text):
    text = text.lower() # lowercase

    y = []
    for i in text.split():
        if i[0] != '@' and i != 'rt': # removing usernames and RT (retweet)
            y.append(i)

    text = ' '.join(y)
    y.clear()
    
    text = nltk.word_tokenize(text) # tokenization

    y = []
    for i in text:
        if i.isalpha(): # removing special characters
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation: # stopwords removal
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i)) # stemming

    return ' '.join(y)

st.title('Inappropriate Content Filtering')

input_text = st.text_area('Enter the text')

if st.button('Predict'):
    transformed_text = transform_text(input_text) # preprocess
    vector_input = tfidf.transform([transformed_text]) # vectorize
    pred = mnb.predict(vector_input)[0] # predict

    if pred == 0:
        st.header('Appropriate content')
    else:
        st.header('Inappropriate content')
