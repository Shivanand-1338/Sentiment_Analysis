🎬 Sentiment Analysis of Movie Reviews using Machine Learning

This project implements Sentiment Analysis on movie reviews using Machine Learning techniques.
The model classifies reviews as Positive or Negative using TF-IDF feature extraction and Logistic Regression.

📌 Project Overview

Sentiment Analysis is a Natural Language Processing (NLP) task that determines the emotional tone behind text.
In this project, we analyze IMDb movie reviews and predict whether the sentiment expressed is positive or negative.

🚀 Features

Uses IMDb dataset from Hugging Face

Text preprocessing using TF-IDF Vectorization

Machine Learning model: Logistic Regression

Model evaluation using:

Accuracy

Precision, Recall, F1-score

Confusion Matrix

Model saving and loading using Pickle

Custom function for predicting sentiment on new text

🛠️ Technologies Used

Python

Scikit-learn

Hugging Face datasets

NumPy

Matplotlib & Seaborn

Google Colab / Jupyter Notebook

📂 Dataset

IMDb Movie Reviews Dataset

50,000 labeled reviews

25,000 training samples

25,000 testing samples

Labels:

0 → Negative

1 → Positive

Dataset Source:
👉 https://huggingface.co/datasets/imdb

⚙️ Workflow

Load Dataset
Load IMDb dataset using Hugging Face datasets.

Text Preprocessing & Feature Extraction

Remove stopwords

Convert text into numerical vectors using TF-IDF

Use unigrams and bigrams

Model Training

Train Logistic Regression classifier on TF-IDF features

Evaluation

Accuracy score

Classification report

Confusion matrix visualization

Model Persistence

Save trained TF-IDF vectorizer and Logistic Regression model using Pickle

Prediction

Load saved model

Predict sentiment for new user input text

📊 Model Performance

Achieves high accuracy on IMDb test data

Logistic Regression performs efficiently for large-scale text classification

TF-IDF captures important textual patterns

(Exact accuracy may vary based on system and subset size.)

🧪 Example Predictions
predict_sentiment_saved("I watched worst movie, great acting!")
# Output: ('NEGATIVE', probability)

predict_sentiment_saved("I loved the movie, great acting!")
# Output: ('POSITIVE', probability)

📁 Project Structure
Sentiment_Analysis/
│
├── Sentiment_Analysis.ipynb
├── models/
│   ├── tfidf.pkl
│   └── logreg.pkl
├── README.md

▶️ How to Run

Clone the repository:

git clone https://github.com/your-username/sentiment-analysis-ml.git


Install dependencies:

pip install datasets transformers evaluate scikit-learn matplotlib seaborn accelerate


Run the notebook:

Sentiment_Analysis.ipynb

🎯 Applications

Movie review analysis

Product review sentiment

Customer feedback analysis

Social media opinion mining

📌 Future Improvements

Add Neutral sentiment classification

Compare with Naive Bayes and SVM

Deploy as Flask/Django web app

Improve performance using deep learning models
