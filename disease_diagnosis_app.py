import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import tensorflow as tf

# Downloading necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('averaged_perceptron_tagger')

model = tf.keras.models.load_model('ANN_Model2.keras')  # Load trained model

# Preprocess class definition as before


class Preprocess:
    @staticmethod
    def remove_punctuations(text):
        if isinstance(text, bytes):
            text = text.decode('utf-8')
        elif isinstance(text, np.ndarray):
            text = text.astype(str)
        removed_punctuations = re.sub(r'[^\w\s]', '', text)
        return removed_punctuations

    @staticmethod
    def tokenizer(text):
        return word_tokenize(text)

    @staticmethod
    def remove_stop_words(tokenized_text):
        stop_words = set(stopwords.words('english'))
        words_filtered = [
            word for word in tokenized_text if word.isalpha() and word not in stop_words]
        return words_filtered

    @staticmethod
    def get_wordnet_pos(word):
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN,
                    "V": wordnet.VERB, "R": wordnet.ADV}
        return tag_dict.get(tag, wordnet.NOUN)

    @staticmethod
    def lemmatize(tokenized_text):
        lemmatizer = WordNetLemmatizer()
        words_filtered = [lemmatizer.lemmatize(
            word, Preprocess.get_wordnet_pos(word)) for word in tokenized_text]
        return ' '.join(words_filtered)

    @staticmethod
    def preprocess_text(text):
        text = text.lower()
        text = Preprocess.remove_punctuations(text)
        tokenized_text = Preprocess.tokenizer(text)
        tokenized_text = Preprocess.remove_stop_words(tokenized_text)
        lemmatized_text = Preprocess.lemmatize(tokenized_text)
        return lemmatized_text


# Mapping the model's output to diagnosis labels
label_mapping = {0: 'Asphyxia', 1: 'Jaundice', 2: 'PREMATURITY', 3: 'SEPSIS'}

# Main function for the Streamlit app


def main():
    # App title and introduction
    st.title("🩺 Disease Diagnosis Assistance")
    st.markdown("""
        ### Welcome to the Disease Diagnosis Assistance App
        This app will help you diagnose common neonatal diseases based on **Symptoms**, **Lab Tests**, and **Treatment**. 
        Please provide the relevant information below.
    """)

    # Sidebar for instructions
    st.sidebar.header("📋 Instructions")
    st.sidebar.info("""
        1. **Symptoms**: Describe the symptoms observed in the patient.
        2. **Lab Tests**: Input the relevant lab test acronyms (e.g., **FBC**, **ESR**).
        3. **Treatment**: Provide any treatments administered so far.
        4. Once all fields are filled, click **Predict Diagnosis**.
    """)
    st.markdown(
        """
        <style>
        div[data-baseweb="base-input"] > textarea {
            min-height: 1px;
            padding: 0;
        }
        </style>
        """, unsafe_allow_html=True
    )

    # Input fields with placeholders and instructions
    st.subheader("Patient Information")
    symptoms = st.text_area("🧑‍⚕️ Symptoms", placeholder="E.g., fever, difficulty breathing, etc.",
                            help="Enter symptoms experienced by the patient.", height=70)
    lab_tests = st.text_area("🧪 Lab Tests (Acronyms)", placeholder="E.g., FBC, ESR",
                             help="Enter lab test results. Please use acronyms.", height=70)

    # Prediction button
    if st.button("🔍 Predict Diagnosis"):
        if symptoms and lab_tests:
            # Combine input into a single sentence
            combined_input = f"{symptoms} {lab_tests}"

            # Preprocess the input text
            preprocessed_input = Preprocess.preprocess_text(combined_input)

            # Predict the diagnosis using the trained model
            prediction = label_mapping[np.argmax(model.predict(
                tf.cast([preprocessed_input], dtype=tf.string)))]
            print(prediction)
            # Display the prediction result
            st.markdown("### Predicted Diagnosis:")
            st.success(f"🩺 The predicted diagnosis is: **{prediction}**")

            # Optionally, provide medical advice or a link to more information
            st.info(
                "For more information about this diagnosis, please consult a healthcare professional.")

        else:
            st.warning(
                "⚠️ Please enter all the fields (Symptoms, Lab Tests, and Treatment) to get a diagnosis.")


# Run the app
if __name__ == '__main__':
    main()
