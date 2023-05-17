# Zillow_project


# Project goal:
Explore the Zillow database and build a Machine Learning model to predict the property tax assessed values of Single Family Properties that had a transaction in 2017.


# Project description:
This project is attempting to predict housing prices according to Zillow data. The desired outcome is to have a functioning ML model that a company can utilize to predict house prices. 

# Initial hypotheses:
- Features most import for property value are:
    - Square feet
    - Number of bedrooms and bathrooms
    - Location
    - Home finishes

# Project planning:
## Wrangle:
- Aquire Zillow database from Codeup MySQL servers
- Prepare data
    - Handle nulls
    - Handle outliers
        - Define parameters to define outliers
    - Encode data if not numeric
    - Split df into train, validate, test
## Explore:
- Using only train data
    - Conduct multivariate analysis to generate 4 questions. Then conduct the following:
        - Hypothesize
        - Visualize
        - Conduct stats test
        - Summarize
    - Question 1
        - Does the number of bathrooms have an impact on the square footage of single family properties?
    - Question 2
        - Does home value increase and decrease respectively with the square footage of a single family property?
    - Question 3
        - Does the number of bathrooms have an impact on the home value of single family properties?
    - Question 4
        - Does the age of the home have an impact on the size of the home (sqft)?
## Scale data:
- Utilize MinMax Scaler before modeling
## Modeling:
- Evaluate models by RMSE and $r^2$
    - Create baseline
    - Ordinary Least Squares
    - LassoLars
    - Polynomial Regression
    - Generalized Linear Model





# Data dictionary:
| Feature | Definition |
|:--------|:-----------|
|bedrooms| The number of bedrooms in a single family residence|
|bathrooms| The number of bathrooms in a single family residence|
|sqft| The amount of square feet in a single family residence|
|property_value| The price a single family residence sold for|
|year_built| The year the single family residence was built|
|county| The county the single family residence is located in|
|rmse| Root Mean Squared Error, accuracy of regression model|
|r2| R-squared (R^2) is a statistical measure that represents the proportion of the variance. 1 = best, 0 = worst|


# Steps to reproduce:
- Clone this repository
- Copy/download all .py files
- Acquire Zillow data from Codeup database using your own env.py file with username and password
- Run zillow-final notebook


# Key findings, recommendations, and takeaways:

- All models performed above baseline
    - But, I would not recommend that any of these models get used in production
- The more features used, the better the models perform
- We may have to build multiple models for this dataset to make more accurate predictions
- We were unable to capture market nuances outside of this data set

# Next Steps:
- Utilize as many features as possible where it makes sense
- Segment data by location within each county
    - Segmenting by county helps but it still doesn't capture the nuances within the county














