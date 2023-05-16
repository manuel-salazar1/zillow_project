#   IMPORTS


import pandas as pd 
import numpy as np

from env import username, password, get_db_url
import os

import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoLars
from sklearn.linear_model import TweedieRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.feature_selection import SelectKBest, RFE, f_regression, SequentialFeatureSelector

import wrangle



# Hypothesis #1 function


def bath_sqft_relationship(train):
    sns.regplot(data=train, x='bathrooms', y='sqft', line_kws={'color':'red'})
    plt.show()
    
    alpha = 0.05
    
    r, p = stats.spearmanr(train.bathrooms, train.sqft)
    print('r =', r)
    print('p =', p)
    
    if p < alpha:
        print("There is a relationship between the number of bathrooms and the square footage of a single family property")
    else:
        print("There is not a relationship between the number of bathrooms and the square footage of a single family property")





# Hypothesis #2 function


def sqft_value_relationship(train):
    sns.regplot(data=train, x='sqft', y='property_value', line_kws={'color':'red'})
    plt.show()
    
    alpha = 0.05

    r, p = stats.spearmanr(train.sqft, train.property_value)
    print('r =', r)
    print('p =', p)
    
    if p < alpha:
        print("There is a relationship between the property value and the square footage of a single family property")
    else:
        print("There is not a relationship between the property value and the square footage of a single family property")



# Hypothesis #3 function

def bathroom_value_relationship(train):
    sns.regplot(data=train, x='bathrooms', y='property_value', line_kws={'color':'red'})
    plt.show()
    
    alpha = 0.05

    r, p = stats.spearmanr(train.bathrooms, train.property_value)
    print('r =', r)
    print('p =', p)
    
    if p < alpha:
        print("There is a relationship between the number of bathrooms and the property value of a single family property")
    else:
        print("There is not a relationship between the number of bathrooms and the property value of a single family property")


# Hypothesis #4 function

def age_sqft_relationship(train):
    sns.regplot(data=train, x='year_built', y='sqft', line_kws={'color':'red'})
    plt.show()
    
    alpha = 0.05
    
    r, p = stats.spearmanr(train.year_built, train.sqft)
    print('r =', r)
    print('p =', p)
    
    if p < alpha:
        print("There is a relationship between the age of a home and the sqft of a single family property")
    else:
        print("There is a relationship between the age of a home and the sqft of a single family property")






# Scale data function

def scale_data(train,
              validate,
              test,
              to_scale):
    '''
    create to_scale variable with a list of columns you want to scale
    returns:
    train_scaled, validate_scaled, test_scaled
    '''
    # make copies for scaling
    train_scaled = train.copy()
    validate_scaled = validate.copy()
    test_scaled = test.copy()
    
    # Make the thing
    scaler = MinMaxScaler()
    
    #fit the thing
    scaler.fit(train[to_scale])
    
    #use the thing
    train_scaled[to_scale] = scaler.transform(train[to_scale])
    validate_scaled[to_scale] = scaler.transform(validate[to_scale])
    test_scaled[to_scale] = scaler.transform(test[to_scale])
    
    return train_scaled, validate_scaled, test_scaled




# X_train, X_validate, X_test, y_train, y_validate, y_test fucntion

def Xy_train_val_test(train, validate, test, target_variable, drop_cols):
    """
    input train, validate, test, after using split function()
    input target_variable as string
    drop_cols formatted as: ['col1', 'col2', 'etc'] for multiple columns
        This function will drop all 'object' columns. Identify additional 
        columns you want to drop and insert 1 column as a string or multiple
        columns in a list of strings.
    returns:
    X_train, X_validate, X_test, y_train, y_validate, y_test
    """
    
    baseline_accuracy = train[target_variable].mean()
    print(f'Baseline Accuracy: {baseline_accuracy}')
    
    X_train = train.select_dtypes(exclude=['object']).drop(columns=[target_variable]).drop(columns=drop_cols)
    X_validate = validate.select_dtypes(exclude=['object']).drop(columns=[target_variable]).drop(columns=drop_cols)
    X_test = test.select_dtypes(exclude=['object']).drop(columns=[target_variable]).drop(columns=drop_cols)
    
    y_train = train[target_variable]
    y_validate = validate[target_variable]
    y_test = test[target_variable]
    
    return X_train, X_validate, X_test, y_train, y_validate, y_test


# evaluate metrics function

def metrics_reg(y, yhat):
    """
    send in y_true, y_pred & returns RMSE, R2
    """
    rmse = mean_squared_error(y, yhat, squared=False)
    r2 = r2_score(y, yhat)
    return rmse, r2



# All models function

def zillow_models(train, X_train, X_validate, y_train, y_validate):
    '''
    Input:
    train, X_trian, X_validate, y_train, y_validate
    Output:
    pred_lr1, pred_pr, pred_glm and model score df
    '''

    # Baseline model, rmse, r2
    baseline = y_train.mean()
    baseline_array = np.repeat(baseline, len(train))
    rmse, r2 = metrics_reg(y_train, baseline_array)
    metrics_df = pd.DataFrame(data=[
        {
            'model': 'baseline',
            'rmse': rmse,
            'r2': r2 
        }
    ])
    
    
    # Ordinary Least Squares (OLS) model
    # make it
    lr1 = LinearRegression()
    # fit it on our RFE features
    lr1.fit(X_train, y_train)
    # use it (make predictions)
    pred_lr1 = lr1.predict(X_train)
    # use it on validate
    pred_validate_lr1 = lr1.predict(X_validate)
    # add to metrics_df
    rmse, r2 = metrics_reg(y_validate, pred_validate_lr1)
    metrics_df.loc[1] = ['Ordinary Least Squares', rmse, r2]


    # LassoLars model
    # make it
    lars = LassoLars(alpha=1, normalize=False)
    # fit it
    lars.fit(X_train, y_train)
    # use it
    pred_lars = lars.predict(X_train)
    pred_validate_lars = lars.predict(X_validate)
    rmse, r2 = metrics_reg(y_validate, pred_validate_lars)
    metrics_df.loc[2] = ['LassoLars', rmse, r2]


    # Polynomial Regression
    # make the polynomial features to get a new set of features
    pf = PolynomialFeatures(degree=2)
    # fit and transform X_train scaled
    X_train_degree2 = pf.fit_transform(X_train)
    # transform X_validate_scaled & X_test_scaled
    X_validate_degree2 = pf.transform(X_validate)
    
    # make it
    pr = LinearRegression()
    # fit it
    pr.fit(X_train_degree2, y_train)
    # use it
    pred_pr = pr.predict(X_train_degree2)
    pred_validate_pr = pr.predict(X_validate_degree2)
    rmse, r2 = metrics_reg(y_validate, pred_validate_pr)
    metrics_df.loc[3] = ['Polynomial Regression', rmse, r2]


    # Generalized Linear Model (GLM)
    # make it
    glm = TweedieRegressor(power=1, alpha=0)
    # fit it
    glm.fit(X_train, y_train)
    # use it 
    pred_glm = glm.predict(X_train)
    pred_validate_glm = glm.predict(X_validate)
    rmse, r2 = metrics_reg(y_validate, pred_validate_glm)
    metrics_df.loc[4] = ['Generalized Linear Model', rmse, r2]
    
    return pred_lr1, pred_pr, pred_glm, metrics_df





# best model function

def best_model(X_train, X_validate, X_test, y_train, y_test):
    # make the polynomial features to get a new set of features
    pf = PolynomialFeatures(degree=2)
    
    # fit and transform X_train scaled
    X_train_degree2 = pf.fit_transform(X_train)
    # transform X_validate_scaled & X_test_scaled
    X_validate_degree2 = pf.transform(X_validate)
    X_test_degree2 = pf.transform(X_test)
    
    # make it
    pr = LinearRegression()
    # fit it
    pr.fit(X_train_degree2, y_train)
    # use it
    pred_pr = pr.predict(X_train_degree2)
    pred_validate_pr = pr.predict(X_validate_degree2)
    pred_test = pr.predict(X_test_degree2)
    
    rmse, r2 = metrics_reg(y_test, pred_test)
    
    return rmse, r2





def actual_pred_plot(pred_lr1, pred_pr, pred_glm, y_train):
    baseline = y_train.mean()
    plt.title('Actual vs. Predicted Values')
   # plt.scatter(pred_lr1, y_train, label='linear regression')
    plt.scatter(pred_pr, y_train, label='ploynominal 2 deg', alpha=.2, s=10, color='gold')
   # plt.scatter(pred_glm, y_train, label='glm', alpha=.1)
    plt.plot(y_train, y_train, label='_nolegend_', color='purple')
    
    plt.axhline(baseline, ls=':', color='grey')
    plt.annotate('Baseline', (0.2, 400_000))
    plt.xlabel('Predicted Values')
    plt.ylabel('Actual Values')
    
    plt.show()






