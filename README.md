# Sentiment Analysis on Amazon Product Reviews

## Overview
This repository contains the project "Sentiment Analysis on Amazon Product Reviews" conducted by Filipa Rodrigues, Mariana Borralho, and Nuno Ferreira, students at ISCTE-IUL. The project aims to explore various text classification methods applied to Amazon product review datasets, utilizing both traditional machine learning models and advanced deep learning techniques.

## Authors
- **Filipa Rodrigues** - ISCTE-IUL, faprs@iscte-iul.pt, no 99865
- **Mariana Borralho** - ISCTE-IUL, msrbo2@iscte-iul.pt, no 120417
- **Nuno Ferreira** - ISCTE-IUL, ntrjf@iscte-iul.pt, no 120557

## Project Description
The project begins with an initial sentiment classification of the test data using pre-trained models from libraries such as TextBlob, Vader Sentiment, and Stanza. After preprocessing tasks aimed at improving the analysis of supervised machine learning classifiers, models like Support Vector Machine and Logistic Regression show promising results with TF-IDF vector representation. Significant gains have been achieved using the Transformer-XL model, optimized for handling long data sequences.

### Key Features
- Utilization of pre-trained sentiment analysis tools (TextBlob, Vader Sentiment, Stanza).
- Advanced text preprocessing techniques including tokenization, stop words removal, stemming, and lemmatization.
- Implementation of machine learning models including SVM, Naïve Bayes, and Logistic Regression.
- Application of Transformer models like Transformer-XL and generative model GPT-3.5 Turbo for enhanced sentiment classification.

## Data Description
The dataset used in this project comprises 48,902 training reviews and 2,417 testing reviews from Amazon, categorized into 'positive' or 'negative' sentiment.

## Repository Structure
 - data/
 - models/ 
 - notebooks/ 
 - src/ 
 - LICENSE
 - README.md 

## Installation
To replicate this analysis, you need to install the required Python libraries:
pip install -r requirements.txt

## Usage
To run the sentiment analysis, navigate to the src directory and execute:
python sentiment_analysis.py

## Results
The project has explored various combinations of preprocessing techniques and models. 
The Transformer-XL model provided the best performance improvement over other models, demonstrating its efficiency in handling long sequences of data relevant in natural language processing tasks.

## Dependencies
 - Pandas
 - NLTK
 - Scikit-learn
 - Spacy
 - TensorFlow
 - PyTorch
 - Transformers

## Contributions
This project welcomes contributions from other students or researchers interested in sentiment analysis.
For major changes, please open an issue first to discuss what you would like to change.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
ISCTE-IUL Faculty Team for guidance and dataset provision.
OpenAI for providing access to GPT models.









This README provides a concise yet comprehensive description of the project, making it easier for other researchers and developers to understand the scope and participate or utilize the project as needed.
