import pandas as pd
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import nltk
import pickle

# Download stopwords (only first time)
nltk.download('stopwords')

print("\nLoading dataset...")
df = pd.read_csv("data/data-set.csv", encoding="latin-1")

df = df[['v1', 'v2']]
df.columns = ['label', 'message']

print("Dataset loaded successfully!")
print(df.head())

ps = PorterStemmer()

def clean_text(msg):
    msg = msg.lower()
    msg = ''.join([char for char in msg if char not in string.punctuation])
    words = msg.split()
    words = [ps.stem(word) for word in words if word not in stopwords.words('english')]
    return ' '.join(words)

print("\nCleaning text messages...")
df['cleaned_message'] = df['message'].apply(clean_text)
print("Text cleaning completed!")

# Feature Extraction (TF-IDF)
print("\nExtracting features using TF-IDF...")
tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(df['cleaned_message']).toarray()

# Labels (ham=0, spam=1)
y = df['label'].map({'ham': 0, 'spam': 1})

# Split Data
print("\nSplitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Data split complete!")

# Train Model
print("\nTraining the Naive Bayes model...")
model = MultinomialNB()
model.fit(X_train, y_train)
print("Model training completed!")

# Evaluate Model
print("\nEvaluating model performance...")
y_pred = model.predict(X_test)

print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save Model & Vectorizer
print("\nSaving model and vectorizer...")
pickle.dump(model, open('models/spam_model.pkl', 'wb'))
pickle.dump(tfidf, open('models/vectorizer.pkl', 'wb'))
print("Model saved as spam_model.pkl and vectorizer.pkl")

# Test With Your Own Email
print("\n--- Test Your Own Email ---")
sample = input("Enter your email/message: ")

vector = tfidf.transform([sample])
prediction = model.predict(vector)

print("\nResult:", "Spam" if prediction[0] == 1 else "Ham (Not Spam)")

print("\nProgram Completed Successfully!")