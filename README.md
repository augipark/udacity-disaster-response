# udacity-disaster-response
Project #2 for Udacity's Data Science Nanodegree Program

## Project Motivation
We want to use data to see what emergency supplies people need in the event of a natural disaster. Different organizations will take care of different parts of the problem. E.g. one organization will care about water, another about blocked roads, another about medical supplies.

Purpose: to output a final model that uses the message column to predict classifications for 36 categories (multi-output classification)

## Python Scripts
### 1. The ETL script: _process_data.py_
Takes the messages.csv and categories.csv files as input and then merges and cleans the dataset to be used for machine learning.
### 2. The machine learning script: _train_classifier.py_
Creates and trains a classifier using the data output from the ETL script.
### 3. The web app: _run.py_
Uses the trained model to display a web app. When a user inputs a message the web app, a classification result is output.
