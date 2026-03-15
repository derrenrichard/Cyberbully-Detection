import pandas as pd
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Load the Dataset
print("Loading dataset...")
df = pd.read_csv('cyberbullying_tweets.csv')

# Clean the Data function
def clean_text(text):
    text = str(text).lower() 
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE) 
    text = re.sub(r'\@\w+|\#', '', text) 
    text = re.sub(r'[^\w\s]', '', text) 
    return text

print("Cleaning text data...")
df['cleaned_text'] = df['tweet_text'].apply(clean_text)

# Split Data
X = df['cleaned_text']
y = df['cyberbullying_type']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize Text
print("Vectorizing text...")
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train Model
print("Training the model...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# Evaluate (Useful for research documentation)
print("\nModel Evaluation:")
y_pred = model.predict(X_test_vec)
print(classification_report(y_test, y_pred))

# Save Model
joblib.dump(model, 'cyberbullying_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
print("Model saved successfully!")