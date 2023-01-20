# udacity-disaster-response
Project #2 for Udacity's Data Science Nanodegree Program

## Project Motivation
We want to use data to see what emergency supplies people need in the event of a natural disaster. Different organizations will take care of different parts of the problem. E.g. one organization will care about water, another about blocked roads, another about medical supplies.

Purpose: to output a final model that uses the message column to predict classifications for 36 categories (multi-output classification).

## Libraries and Installation
- import sys
- from sqlalchemy import create_engine
- import nltk
- nltk.download('punkt')
- nltk.download('wordnet')
- import re
- import numpy as np
- import pandas as pd
- import pickle
- from sklearn.model_selection import GridSearchCV
- from nltk.tokenize import word_tokenize
- from nltk.stem import WordNetLemmatizer
- from sklearn.metrics import confusion_matrix
- from sklearn.model_selection import train_test_split
- from sklearn.ensemble import RandomForestClassifier
- from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
- from sklearn.pipeline import Pipeline
- from sklearn.multioutput import MultiOutputClassifier
- from sklearn.metrics import classification_report

## Python Scripts
### 1. The ETL script: _process_data.py_
Takes the messages.csv and categories.csv files as input and then merges and cleans the dataset to be used for machine learning.
### 2. The machine learning script: _train_classifier.py_
Creates and trains a classifier using the data output from the ETL script.
### 3. The web app: _run.py_
Uses the trained model to display a web app. When a user inputs a message the web app, a classification result is output.

## File Descriptions 
1. README.md - read me file
2. messages.csv - input dataset #1
3. categories.csv - input dataset #2
4. ETL Pipeline Preparation.ipynb - rough work to create process_data.py
5. process_data.py - see "Python Scripts" above
6. DisasterResponse.db - output from process_data.py
7. ML Pipeline Preparation.ipynb - rough work to create train_classifier.py
8. train_classifier.py - see "Python Scripts" above
9. classifier.pkl - output pickle file from train_classifier.py
10. run.py - see "Python Scripts" above
## Instructions
