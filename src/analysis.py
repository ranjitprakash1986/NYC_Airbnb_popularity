# Author: Ranjit Sundaramurthi
# Date:  2022-12-21


"""
Script to read the train data, create pipelines to transformations to scale it.
Run Models on the training data.

Usage: src/analysis.py --input_file=<input_file>

Options:
--input_file=<input_file>       Input training file with complete path

Example:
>>>python src/analysis.py --input_file="data/processed/train.csv"
"""

# imports
from docopt import docopt

import pandas as pd
import numpy as np
import os
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.compose import (
    ColumnTransformer,
    TransformedTargetRegressor,
    make_column_transformer,
)
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.linear_model import LassoCV
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
from sklearn.model_selection import (
    GridSearchCV,
    cross_val_score,
    cross_validate,
    train_test_split,
)

opt = docopt(__doc__)

def transform(input_file):
    """
    Function that takes the inputted the train dataframe and creates the transformers for scaling the data.
    
    Parameters:
    -----------
    input_file:        train data file
    
    Returns:
    ----------
    preprocessor:      column transformer object
    """
    
   
    # Read the dataframe
    df = pd.read_csv(input_file)
    
    # Numerica columns
    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
    
    # Remaining Columns
    categorical_features = ['neighbourhood', 'neighbourhood_group', 'room_type', 'host_name_mod']
    text_feature = "name"
    drop_features = ["host_name"]
    

    #checking that the count of columns matches with the dataframe ## 1 for 1 text feature
    if not assert(len(numeric_features) + len(categorical_features) + len(drop_features) + 1) == len(df.columns):
        print("Mismatch in total feature column count and the total columns in dataframe")
        
        
    # Transformations, adding imputations if needed later in the the process

    numeric_transformer = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())

    categorical_transformer = make_pipeline(
        SimpleImputer(strategy="constant", fill_value="missing"),
        OneHotEncoder(handle_unknown="ignore", sparse = False)
    )


    preprocessor = make_column_transformer(
        ("drop", drop_features),
        (numeric_transformer, numeric_features),
        (categorical_transformer, categorical_features),
        (CountVectorizer(stop_words="english"), text_feature),
    )
    
    return preprocessor

# function adopted from 571
def mean_std_cross_val_scores(model, X_train, y_train, **kwargs):
    """
    Returns mean and std of cross validation

    Parameters
    ----------
    model :
        scikit-learn model
    X_train : numpy array or pandas DataFrame
        X in the training data
    y_train :
        y in the training data

    Returns
    ----------
        pandas Series with mean scores from cross_validation
    """

    scores = cross_validate(model, X_train, y_train, **kwargs)

    mean_scores = pd.DataFrame(scores).mean()
    std_scores = pd.DataFrame(scores).std()
    out_col = []

    for i in range(len(mean_scores)):
        out_col.append((f"%0.3f (+/- %0.3f)" % (mean_scores[i], std_scores[i])))

    return pd.Series(data=out_col, index=mean_scores.index)
   

    
# function to get performance for different model
def model_score(model, X_train, X_test, y_train, y_test):
    """
    Returns the fitted model and the list of train and test scores.

    Parameters:
    ------
    model: (obj)
    the ML molel
    
    X_train: (dataframe)
    the train dataset
    
    X_test: (dataframe)
    the test dataset
    
    y_train: (series)
    the target variable of train data

    y_test: (series)
    the target variable of test data

    Returns:
    -------
    model: (obj)
    the model object fitted on the train data
    
    results: (list)
    the R2 and rmse scores on the train and test data
    """
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    r2_score_train = r2_score(y_train, y_train_pred)
    r2_score_test = r2_score(y_test, y_test_pred)
    
    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    results = [r2_score_train, r2_score_test, rmse_train, rmse_test]
    
    return model, results


def showResults(results, name):
    """
    Prints the results computed by model_score() function.

    Parameters:
    ------
    results: (list)
    list of R2 and rmse scores
    
    Returns:
    -------
    None: 
    No value returned
    """
    r2_score_train, r2_score_test, rmse_train, rmse_test = results
    lst = [name + '_r2_score_train_', name+ '_r2_score_test', name+'_rmse_train', name+'_rmse_test']
    df_result = pd.DataFrame(list(zip(lst, results)),
               columns =['Score_Name', 'Score'])
    
    print('Results:')
    print('R2 score on training set: {:.3f}'.format(r2_score_train))
    print('R2 score on testing set: {:.3f}'.format(r2_score_test))
    print('RMSE on training set: {:.3f}'.format(rmse_train))
    print('RMSE on testing set: {:.3f}'.format(rmse_test))
    return df_result