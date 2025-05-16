# Sentiment_Analysis

This project applies a fine-tuned DistilBERT model for binary sentiment classification of IMDB movie reviews. The goal is to distinguish between positive and negative sentiments with high accuracy and computational efficiency. This project was developed as part of the Machine Learning 2 course in the Department of Physics at the University of Crete.

## Features

- **Model:** DistilBERT (fine-tuned using Hugging Face Transformers)  
- **Dataset:** Subset of 7,000 balanced IMDB reviews  
- **Accuracy:** 88.7%  
- **F1 Score:** 88.8%  
- **Deployment:** Streamlit web app for real-time predictions  

## How to Use

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
2. Run the Streamlit app in the last cell of the Notebook
   ```bash
   streamlit run streamlit_bert.py
4. Enter a movie review or whatever you please in the interface to get its sentiment and confidence score.

Author: Georgios Kritopoulos

Course: Machine Learning 2 (PH253), University of Crete
