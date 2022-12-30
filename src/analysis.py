# Author: Ranjit Sundaramurthi
# Date:  2022-12-21


"""
Script to read the train data, create pipelines to transformations to scale it.
Run Models on the training data.

Usage: src/analysis.py --train_file=<train_file> --test_file=<test_file> --output_dir=<output_dir>

Options:
--train_file=<train_file>       Training file with complete path
--test_file=<test_file>         Test file with complete path
--output_dir=<output_dir>       Output directory of the analysis

Example:
>>>python src/analysis.py --train_file="data/processed/train.csv" --test_file="data/processed/test.csv" --output_dir="results/"
"""

# imports
from docopt import docopt

import pandas as pd
import numpy as np
import os
import pickle
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.compose import (
    ColumnTransformer,
    TransformedTargetRegressor,
    make_column_transformer,
)
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Lasso
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
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import RandomizedSearchCV
import random
from scipy.stats import randint

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

opt = docopt(__doc__)

def main(train_file, test_file, out_dir):
    '''
    Run hyperparameter optimization using R2 score on linear models and non-linear models,
    Perform model training & saving
    
    Parameters
    ----------
    train_file : str
        training data file with complete path

    out_dir: str
        output directory of the analysis
     
    Returns
    -------
    None 
        

    Examples
    --------
    >>> main(opt["--train"], opt["--output_dir"])
    '''
    
    print("Begin training models")
    print("="*100)
    
    
    
    # get training data
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    
    X_train, y_train = train_df.drop(columns=["reviews_per_month"]), train_df["reviews_per_month"]
    X_test, y_test = test_df.drop(columns=["reviews_per_month"]), test_df["reviews_per_month"]
    
    # Execute the all models
    allmodels(X_train, y_train, out_dir)
    
    # Hyperparameter tuning lgbm models
    param_tuning(X_train, y_train, out_dir)
    
    # Best model score
    modelname = 'lgbmregressor'
    with open(out_dir + 'model_' + modelname ,'rb') as file:
        model = pickle.load(file)
    
    results = model_score(model, X_train, X_test, y_train, y_test)
    
    # print the scores
    tabulateresults(results, modelname, out_dir)
        
def allmodels(X_train, y_train, out_dir):
    
    # initialize our models
    results_dict = {}
    alphas = 10.0 ** np.arange(-6, 6, 1)
    
    linear_models = {
        "linearregression": LinearRegression(),
        "ridgecv":RidgeCV(alphas=alphas, cv=10),
        "lasso": Lasso(alpha=0.1),
        "decision_tree": DecisionTreeRegressor(),
        "random_forest":RandomForestRegressor(),
        "GradientBoosting": GradientBoostingRegressor(),
        "lgbm": LGBMRegressor(),
        "xgboost": XGBRegressor()
    }
                                     
    # for each model 
    for model_name, model in linear_models.items():
        print("Training "+model_name)
        pipe = make_pipeline(transformer(X_train), model)
        results_dict[model_name] = mean_std_cross_val_scores(
            pipe,
            X_train,
            y_train,
            cv=10,
            n_jobs = -1,
            return_train_score=True
        )
        
    # save the fitted model
        try:
            file_log = open(out_dir + '/model_' + model_name, 'wb')
        except:
            os.makedirs(os.path.dirname(out_dir))
            file_log = open(out_dir + '/model_' + model_name, 'wb')
        pickle.dump(pipe, file_log)

    print("Training linear models complete")
    print("="*100)
    
    # Results dataframe
    results_df = pd.DataFrame(results_dict).T
    
    # Save the results to a csv file
    print('Writing the train and cross validation results file')
    print('='*100)
    results_df.to_csv(out_dir + 'allmodel_train_results.csv')
    print('Writing results completed')
    print('='*100)
    
        
def param_tuning(X_train, y_train, out_dir):
    
    # Define the parameters and their range values to tune
    
    param_dist = {  
    "lgbmregressor__num_leaves": [8, 25, 50, 75, 100, 150, 200, 350, 500],
    "lgbmregressor__max_depth": [1,3,5,7,9],
    "lgbmregressor__learning_rate": [0.01,0.1,1,10],
    "lgbmregressor__n_estimators": [750, 1000, 1200]
    }
    
    # model
    lgbm_pipe = make_pipeline(transformer(X_train), LGBMRegressor())
    
    # random search for best parameters
    print("Begin hyperparameter tuning")
    random_search_lgbm = RandomizedSearchCV(
    lgbm_pipe,
    param_distributions=param_dist,
    n_iter=10,
    verbose=1,
    n_jobs=-1,
    cv=5,
    random_state=123
    )

    random_search_lgbm.fit(X_train, y_train)
    
    # Output the best parameters to csv file
    best_parameters_lgbm = pd.DataFrame(random_search_lgbm.best_params_.items(), columns=['hyperparameter', 'Value'])
    print('Writing the best parameters to results folder')
    print('='*100)
    best_parameters_lgbm.to_csv(out_dir + 'best_parameters_lgbm.csv')
    print('Writing results completed')
    print('='*100)
    
    # Running the best model
    results_dict = {}
    results_dict["best_lgbm"] = mean_std_cross_val_scores(
        random_search_lgbm.best_estimator_,
        X_train,
        y_train,
        cv=5,
        n_jobs =-1,
        return_train_score=True
    )
    
    # Exporting the best_lgbm train model and results
    file_log = open(out_dir + '/model_' + 'lgbmregressor', 'wb')
    pickle.dump(random_search_lgbm.best_estimator_, file_log)
    
    results_df = pd.DataFrame(results_dict).T
    results_df.to_csv(out_dir + 'lgbmregressor_training_results.csv')
    

def transformer(input_var):
    """
    Function that takes the inputted the train dataframe and creates the transformers for scaling the data.
    
    Parameters:
    -----------
    input_var:        X_train variable
    
    Returns:
    ----------
    preprocessor:      column transformer object
    """
    
   
    # Assign the variable to a dataframe
    df = input_var
    
    # Numerica columns
    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
    
    # Remaining Columns
    numeric_features = numeric_columns
    categorical_features = ['neighbourhood', 'neighbourhood_group', 'room_type', 'host_name_mod']
    text_feature = "name"
    drop_features = ["host_name"]
    

    #checking that the count of columns matches with the dataframe ## 1 for 1 text feature + 1 for the target column
    # if (assert(len(numeric_features) + len(categorical_features) + len(drop_features) + 1 + 1) == len(df.columns)):
    #     pass
    # else:
    #     print("Mismatch in total feature column count and the total columns in dataframe")
        
        
    # Transformations, adding imputations if needed later in the the process

    numeric_transformer = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())

    categorical_transformer = make_pipeline(
        SimpleImputer(strategy="constant", fill_value="missing"),
        OneHotEncoder(handle_unknown="ignore", sparse = False)
    )


    return_object = make_column_transformer(
        ("drop", drop_features),
        (numeric_transformer, numeric_features),
        (categorical_transformer, categorical_features),
        (CountVectorizer(stop_words="english"), text_feature),
    )
    
    return return_object

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
    
    return results


def tabulateresults(results, name, out_dir):
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
    
    df_result.to_csv(out_dir + 'model_scores.csv')
    
    # print('Results:')
    # print('R2 score on training set: {:.3f}'.format(r2_score_train))
    # print('R2 score on testing set: {:.3f}'.format(r2_score_test))
    # print('RMSE on training set: {:.3f}'.format(rmse_train))
    # print('RMSE on testing set: {:.3f}'.format(rmse_test))
    return df_result


if __name__ == "__main__":
    main(opt["--train_file"], opt["--test_file"],opt["--output_dir"])