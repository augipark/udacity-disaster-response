import sys
# import libraries
from sqlalchemy import create_engine
import nltk
nltk.download('punkt')
nltk.download('wordnet')

import re
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import GridSearchCV
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report

def load_data(database_filepath):
    # load data from database
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('disasterdata', engine)
    # X should be the message column and Y should be all of the other columns
    X = np.ravel(df[["message"]].values)
    column_names = list(df.iloc[:,4:100].columns)
    y = df.iloc[:, 4:100].values
    
    return X, y, column_names

regex_url = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

def tokenize(text):
    url_detect = re.findall(regex_url, text)
    for url in url_detect:
        text = text.replace(url, "place_holder_url")

    words = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_words = []
    for word in words:
        clean_word = lemmatizer.lemmatize(word).lower().strip()
        clean_words.append(clean_word)

    return clean_words


def build_model():
    pipeline = Pipeline([
    ('vectorizer', CountVectorizer(tokenizer=tokenize)),
    ('tefreq_indocfreq', TfidfTransformer()),
    ('classifier', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {
    'classifier__estimator__n_estimators': [5, 10],
    'classifier__estimator__min_samples_split': [2, 3]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=3, n_jobs=8, cv=3)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    y_pred_best = model.predict(X_test)
    
    for i in range(len(category_names)):
        print(category_names[i])
        print(classification_report(Y_test[:,i], y_pred_best[:,i]))


def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()