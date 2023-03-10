import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    load_data
    loads data from 2 csv files and merges them into a single dataframe.
    
    Input:
    messages_filepath   filepath to the messages csv file
    categories_filepath filepath to the categories csv file
    
    Output:
    df_new      merged dataframe of the messages and categories files
    '''
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    
    # merge datasets
    df = pd.merge(messages, categories, on="id")
    
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(";", expand=True)
    
    row = df['categories'].str.split(";", expand=True).iloc[0]
    category_colnames = row.apply(lambda x: x.rstrip('-0 -1'))
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        print(categories[column])
        categories[column] = categories[column].str[-1]


        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
        
    categories.loc[categories["related"] == 2, "related"] = 1
    
    # drop the original categories column from `df`
    df = df.drop(columns=['categories'])
    
    # concatenate the original dataframe with the new `categories` dataframe
    df_new = pd.concat([df, categories], axis=1)
    
    return df_new


def clean_data(df):
    '''
    clean_data
    drops duplicates in the dataframe
    
    Input:
    df           the merged dataframe output by the load_data function
    
    Output:
    df_new2      merged dataframe without duplicates
    '''
    # drop duplicates
    df_new2 = df.drop_duplicates()
    
    return df_new2


def save_data(df, database_filename):
    '''
    save_data
    save the merged and cleaned dataframe into an sqlite database
    
    Input:
    df           the merged and cleaned dataframe output by the clean_data function
    database_filename   the filename you want to give the sqlite database
    
    Output:
    nothing is returned, but the result is we write records store in a dataframe to a sqlite database
    '''
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('disasterdata', engine, if_exists='replace', index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
