# Sentiment Analysis Prediction

A machine learning-based sentiment analysis dashboard designed to predict whether a Gojek user review expresses a positive or negative sentiment. This project demonstrates the use of Natural Language Processing, machine learning, and a simple interactive web dashboard built with Streamlit.

## Project Overview

This project was developed to help analyze customer feedback more efficiently. Instead of reading large numbers of reviews manually, the system allows users to input a review text and receive an automatic sentiment prediction.

The dashboard focuses on classifying user comments into two sentiment categories:

- Positive
- Negative

This project reflects practical computer science skills in data preprocessing, Natural Language Processing, machine learning model deployment, and user-facing application development.

## Key Features

- Interactive sentiment prediction dashboard
- Text input for customer review analysis
- Sentiment classification into positive or negative categories
- Confidence score display when model probability is available
- Machine learning model integration using saved model files
- Streamlit-based web application interface
- TF-IDF vectorizer and Logistic Regression model support
- LSTM model artefact included for deep learning-based sentiment analysis

## Tech Stack

This project uses the following technologies:

- Python
- Streamlit
- NumPy
- Pandas
- Scikit-Learn
- TensorFlow
- Keras
- Joblib
- NLTK
- Sastrawi

## Repository Structure

```text
SentimentAnalysisPrediction/
│
├── app.py
├── logistic_regression_model.joblib
├── lstm_model.keras
├── tfidf_vectorizer.joblib
├── tokenizer.pickle
├── requirements.txt
└── README.md
