import streamlit as st
import pickle
import os

# Load Model and Vectorizer

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'spam_model.pkl')
VECTORIZER_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'vectorizer.pkl')

try:
    with open(MODEL_PATH, 'rb') as model_file:
        model = pickle.load(model_file)

    with open(VECTORIZER_PATH, 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)

except FileNotFoundError:
    st.error("Model or Vectorizer file not found! Please train your model first using 'spam_classifier.py'.")
    st.stop()

# Streamlit UI
st.set_page_config(page_title="MailGuard AI - Spam Classifier", page_icon="📧", layout="centered")

st.title("MailGuard AI - Email Spam Classifier")
st.write("This intelligent system uses **Machine Learning** to detect whether an email is **Spam** or **Not Spam (Ham)**.")

email_text = st.text_area("Enter your email content below:", height=200, placeholder="Type or paste your email message here...")

if st.button("🔍 Analyze Email"):
    if not email_text.strip():
        st.warning("Please enter some email text.")
    else:
        # Convert text to vector
        input_data = vectorizer.transform([email_text])
        
        # Predict
        prediction = model.predict(input_data)[0]

        # Display result
        if prediction == 1:
            st.error("This email is **SPAM!** Be cautious before clicking any links.")
        else:
            st.success("This email is **NOT SPAM (HAM)**. It seems safe.")
        
        # Optional confidence score
        try:
            prob = model.predict_proba(input_data)[0]
            spam_prob = prob[1] * 100
            st.progress(spam_prob / 100)
            st.caption(f"Spam probability: **{spam_prob:.2f}%**")
        except Exception:
            pass

st.markdown("---")
st.caption("Developed by **Vivek Kushwaha**")