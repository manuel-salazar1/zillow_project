# Imports

import pandas as pd 
from env import username, password, get_db_url
import os
from sklearn.model_selection import train_test_split
import seaborn as sns
import numpy as np



#-------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------

# Path to read in data from MySQL

def new_zillow_data():
    """
    This function will:
    - create a connect_url to mySQL
    - return a df of the given query from the zillow db
    """
    url = get_db_url('zillow')
    SQL_query = '''
                select bedroomcnt as bedrooms, bathroomcnt as bathrooms, calculatedfinishedsquarefeet as sqft
                    , taxvaluedollarcnt as property_value, yearbuilt as year_built, fips as county
                from properties_2017
                join predictions_2017 using(parcelid)
                where propertylandusetypeid = 261
                '''
    return pd.read_sql(SQL_query, url)



#-------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------


# Path to either create new CSV or to Find existing CSV

def get_zillow_data(filename="zillow.csv"):
    """
    This function will:
    - Check local directory for csv file
        - return if exists
    - If csv doesn't exists:
        - create a df of the SQL_query
        - write df to csv
    - Output zillow df
    """
    if os.path.exists(filename):
        df = pd.read_csv(filename, index_col=0) 
        print('Found CSV')
        return df
    
    else:
        df = new_zillow_data()
        
        #want to save to csv
        df.to_csv(filename)
        print('Creating CSV')
        return df




#-------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------


# CLEANING DATA

def wrangle_zillow(df):
    '''
    This function is:
    - Dropping nulls
    - Changing dtype to int for bedrooms, sqft, tax_value, year_built
    - Assigning county names
    - Handling outliers for:
         bedrooms, bathrooms, sqft, property_value
    Returns:
    df
    '''
    
    df = df.copy()
    df = df.dropna()
    
    make_ints = ['bedrooms', 'sqft', 'property_value', 'year_built']

    for col in make_ints:
        df[col] = df[col].astype(int)
        
    df.county = df.county.map({6037:'la', 6059:'orange', 6111:'ventura'})
    
    df = df[df.bedrooms.between(1,7)]
    df = df[df.bathrooms.between(1,6)]
    df = df[df.sqft <= df.sqft.mean() + (3 * df.sqft.std())]
    df = df[df.sqft >= 500]
    df = df[df.property_value < df.property_value.quantile(.95)]
    df = df[df.property_value > 100_000]
    
    return df



#-------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------


# This function will create dummy variable columns

dummy_cols = 'county'

def create_dummy_variables(df):
    '''
    inputs:
    df , variable with list of strings or single string
    output:
    df with dummy columns
    '''
    dummy_df = pd.get_dummies(df[dummy_cols], drop_first=True)
    df = pd.concat([df, dummy_df], axis=1)
    return df



#-------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------



# SPLIT FUNCTION

def split_function(df):
    '''
    Take in a data frame and returns:
    train, validate, test 
    subset data frames
    '''
    train, test = train_test_split(df,
                              test_size=0.20,
                              random_state=123,
                                  )
    train, validate = train_test_split(train,
                                  test_size=.25,
                                  random_state=123,
                                      )
    return train, validate, test








