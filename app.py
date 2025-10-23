import streamlit as st
import pickle
import os
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')

# Load model and vectorizer directly from models folder
model = pickle.load(open('models/spam_model.pkl', 'rb'))
vectorizer = pickle.load(open('models/vectorizer.pkl', 'rb'))

ps = PorterStemmer()

def clean_text(msg):
    msg = msg.lower()
    msg = ''.join([c for c in msg if c not in string.punctuation])
    words = msg.split()
    words = [ps.stem(word) for word in words if word not in stopwords.words('english')]
    return ' '.join(words)

st.title("MailGuard AI - Email Spam Detector")
st.write("Type your email or message below and check if it's spam or not!")

input_msg = st.text_area("Enter email/message:")

if st.button("Analyze"):
    if not input_msg.strip():
        st.warning("Please enter a valid message or email.")
    else:
        cleaned = clean_text(input_msg)
        vector = vectorizer.transform([cleaned])
        prediction = model.predict(vector)[0]
        proba = model.predict_proba(vector)[0][1] * 100

        if prediction == 1:
            st.error(f"Spam Detected! (Spam Probability: {proba:.2f}%)")
        else:
            st.success(f"Not Spam (Spam Probability: {proba:.2f}%)")

st.caption("Developed by Vivek Kushwaha")