import streamlit as st
import joblib
import re

# Load the saved model and vectorizer
model = joblib.load('cyberbullying_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text

# Streamlit User Interface
st.set_page_config(page_title="Cyberbullying Detector", page_icon="🛡️")
st.title("🛡️ Cyberbullying Detection App")
st.write("Enter a sentence below to test the detection model.")

user_input = st.text_area("Enter text here:", height=150)

if st.button("Analyze Text"):
    if user_input.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        # Process the input just like the training data
        cleaned_input = clean_text(user_input)
        vectorized_input = vectorizer.transform([cleaned_input])
        
        # Predict
        prediction = model.predict(vectorized_input)[0]
        
        # Show Results
        st.subheader("Analysis Result:")
        if prediction == "not_cyberbullying":
            st.success("✅ This text does not appear to contain cyberbullying.")
        else:
            st.error(f"⚠️ Warning: Detected **{prediction.upper()}** cyberbullying.")