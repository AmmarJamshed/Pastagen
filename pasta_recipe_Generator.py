#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from collections import Counter

# Download necessary NLTK data
nltk.download('vader_lexicon')
nltk.download('stopwords')

# Initialize sentiment analyzer
sid = SentimentIntensityAnalyzer()

# Sample data
data = {
    "customer_id": [1, 2, 3, 4, 5],
    "feedback": [
        "I love the spicy pasta. It has the perfect amount of heat and flavor.",
        "The creamy pasta was too bland for my taste. Needs more seasoning.",
        "Great texture on the al dente pasta, but the sauce was too sweet.",
        "The seafood pasta was delicious. I especially enjoyed the garlic butter sauce.",
        "The pasta was overcooked and mushy. The sauce lacked depth."
    ]
}

# Create DataFrame
feedback_df = pd.DataFrame(data)

# Function to analyze feedback
def analyze_feedback(feedback_df):
    feedback_df['sentiments'] = feedback_df['feedback'].apply(lambda x: sid.polarity_scores(x))

    # Extract most common words excluding stopwords
    stop_words = set(stopwords.words('english'))
    all_words = ' '.join(feedback_df['feedback']).lower().split()
    filtered_words = [word for word in all_words if word not in stop_words]
    word_freq = Counter(filtered_words)

    return feedback_df, word_freq

# Function to suggest recipes
def suggest_recipe(feedback_df, word_freq):
    negative_feedback = feedback_df[feedback_df['sentiments'].apply(lambda x: x['compound']) < 0]
    positive_feedback = feedback_df[feedback_df['sentiments'].apply(lambda x: x['compound']) > 0]

    improvements = []
    for feedback in negative_feedback['feedback']:
        if 'bland' in feedback or 'seasoning' in feedback:
            improvements.append('Add more seasoning')
        if 'overcooked' in feedback:
            improvements.append('Cook pasta for a shorter time')
        if 'sweet' in feedback:
            improvements.append('Reduce the sweetness of the sauce')

    popular_ingredients = [word for word, count in word_freq.most_common() if word not in stop_words and word not in improvements]

    return {
        'improvements': improvements,
        'popular_ingredients': popular_ingredients[:5]
    }

# Streamlit interface
st.title('Pasta Feedback Analyzer')
st.write("This app analyzes customer feedback on pasta taste and suggests improvements for new recipes.")

feedback_df, word_freq = analyze_feedback(feedback_df)

if st.checkbox('Show raw data'):
    st.subheader('Raw Data')
    st.write(feedback_df)

if st.checkbox('Show sentiment analysis'):
    st.subheader('Sentiment Analysis')
    st.write(feedback_df[['feedback', 'sentiments']])

if st.checkbox('Show word frequency'):
    st.subheader('Word Frequency')
    st.write(word_freq.most_common(10))

recipe_suggestions = suggest_recipe(feedback_df, word_freq)

st.subheader('Recipe Suggestions')
st.write("### Improvements")
for improvement in recipe_suggestions['improvements']:
    st.write(f"- {improvement}")

st.write("### Popular Ingredients")
for ingredient in recipe_suggestions['popular_ingredients']:
    st.write(f"- {ingredient}")


