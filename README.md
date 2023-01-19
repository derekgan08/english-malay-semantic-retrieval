# Natural Language Processing Assignment 2: English-Malay Semantic Retrieval

## Problem Statements
In bilingual or multilingual environments, efficiently determining the semantic similarity between sentences in different languages (e.g., English and Malay) is important for tasks such as machine translation, cross-lingual information retrieval, and language understanding. This assignment addresses the challenge of classifying sentence similarity by leveraging deep learning techniques for natural language processing (NLP).

## Project Overview
This assignment utilizes a deep learning approach to classify the similarity between English and Malay sentences. The goal is to predict whether two sentences, one in English and one in Malay, have the same meaning. The assignment is built using LSTM (Long Short-Term Memory) networks, which are well-suited for sequential data, and pre-trained GloVe embeddings to represent the words semantically.

The neural network is designed with two models: one for processing English sentences and another for processing Malay sentences. After embedding the sentences using GloVe, the output from both models is concatenated and passed through a fully connected layer to predict binary similarity (0 or 1). The model is trained on a labeled dataset and evaluated on both a validation and a test set to assess its performance.

## Key Features
- **Dual-Language Model**: Handles English and Malay sentences simultaneously.
- **GloVe Embeddings**: Uses pre-trained GloVe embeddings to represent words semantically.
- **LSTM Network**: Uses an LSTM-based architecture to handle the sequential nature of language.
- **Binary Classification**: The model classifies sentence pairs as either similar (1) or not similar (0).
- **Model Evaluation**: The model is evaluated on both validation and test datasets to measure performance.

## Technologies Used
- **Python**
- **TensorFlow**
- **Keras**
- **NumPy**
- **GloVe Pre-trained Embeddings**
- **Matplotlib** (for visualizing the model’s performance)

## References
1. [Keras Documentation](https://keras.io/api/)

2. [Pre-trained Word embedding using Glove in NLP models](https://www.geeksforgeeks.org/pre-trained-word-embedding-using-glove-in-nlp-models/)

3. [Recurrent Neural Networks by Example in Python](https://towardsdatascience.com/recurrent-neural-networks-by-example-in-python-ffd204f99470)