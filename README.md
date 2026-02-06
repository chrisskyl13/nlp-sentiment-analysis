# Amazon Reviews Sentiment Analysis: Comparison of 4 NLP Approaches

This project provides a comprehensive comparison of different Natural Language Processing (NLP) techniques for sentiment analysis using Amazon product reviews.

##  Overview
The goal is to classify reviews as positive or negative by evaluating four distinct methodologies on the same dataset:

1. **Word2Vec**: Training custom embeddings from scratch combined with a Logistic Regression classifier.
2. **GloVe**: Utilizing pre-trained 100-dimensional embeddings from Stanford (via Gensim).
3. **BERT (Feature Extraction)**: Using `bert-base-uncased` to extract static embeddings and classifying them with Logistic Regression.
4. **BERT (Fine-Tuning)**: End-to-end training of a `distilbert-base-uncased` model using TensorFlow/Keras for optimal performance.

##  Technologies Used
* **Python**
* **TensorFlow / PyTorch**
* **Hugging Face Transformers**
* **Scikit-learn**
* **Gensim** (Word2Vec & GloVe)
* **NLTK** (Tokenization & Cleaning)

##  Key Features
* Automated text cleaning and label normalization.
* Performance evaluation using Accuracy and F1-Score.
* Visualization of results through Confusion Matrices.
* Comparative metrics table for all four approaches.
