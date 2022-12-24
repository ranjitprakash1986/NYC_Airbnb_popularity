# Author: Ranjit Sundaramurthi
# Date:  2022-12-21

"""
Script to process the raw data and split into train and test
These are stored in data/processed folder

Usage: src/preprocessing.py --input_file=<input_file> --output_path=<output_path>

Options:
--input_file=<input_file>   Input file which represents the raw data
--output_path=<output_path> Path where the output processed data is to be stored

Example:
>>>python src/preprocessing.py --input_file="data/raw/data.csv" --output_path="data/processed/"
"""


# imports

from docopt import docopt
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.compose import (
    ColumnTransformer,
    TransformedTargetRegressor,
    make_column_transformer,
)

opt = docopt(__doc__)


def preprocess(df, output_path):
    """
    Perform the processing of raw data
    
    Parameters:
    ----------
    df: Pandas Dataframe that contains the raw data

    Returns:
    -------
    df: Pandas Dataframe with processed data

    Examples
    --------
    >>> preprocess(df)
    df object
    """

    # Preprocessing the data

    # drop rows with the missing values
    df = df[df["name"].notna()]
    df = df[df["host_name"].notna()]
    df = df[df["last_review"].notna()]
    df = df[df["reviews_per_month"].notna()]

    # change dates of review to pandas inbuilt date
    df["last_review"] = pd.to_datetime(df["last_review"], infer_datetime_format=True)

    # Create a new feature column "days_since_review"
    df = df.assign(
        days_since_review=df["last_review"].apply(
            lambda x: (np.datetime64("today") - x).days
        )
    )

    # Now we can drop the last_review column
    # as we have captured its effect through days_since_review
    # Dropping host_id since it has unique identifier for most rows
    df = df.drop(columns=["last_review", "host_id", "id"])

    # Suggestion for feature transformation to reduce the number of categories
    # to reduce the number of unique categories for host_name we could bunch together under "other"
    name_df = pd.DataFrame(df["host_name"].value_counts())

    # Determining the threshold for which roughly half of the data set is "other"
    threshold = 14  # this moves fairly 50% to other category

    name_other = name_df[name_df["host_name"] < threshold].index.tolist()
    df = df.assign(
        host_name_mod=df["host_name"].apply(lambda x: "other" if x in name_other else x)
    )
    
    # Storing the processed data
    try:
        df.to_csv(output_path + "processed_data.csv", index=False)
    except:
        os.makedirs(os.path.dirname(output_path))
        df.to_csv(output_path + "processed_data.csv", index=False)
        
    # Display message
    print(
        "Raw data was processed and saved in destination folder"
    )
    

#     # Splitting the data
#     train_df, test_df = train_test_split(df, test_size=test_size, random_state=573)

#     # Storing the data
#     try:
#         train_df.to_csv(output_path + "train_eda.csv", index=False)
#         test_df.to_csv(output_path + "test.csv", index=False)
#     except:
#         os.makedirs(os.path.dirname(output_path))
#         train_df.to_csv(output_path + "train.csv", index=False)
#         test_df.to_csv(output_path + "test.csv", index=False)

#     # Display message
#     print(
#         "Raw data was processed and split into train and test data in destination folder"
#     )


def main(input_file, output_path):
    """
    Main function to tie all processing together

    Parameters:
    ----------

    Returns:
    -------

    """

    # Read the data
    print("Reading raw data")
    df = pd.read_csv(input_file)
    print("Reading raw data completed\n" + "=" * 50)

    # Call the function for preprocessing
    print("Preprocessing starting")
    preprocess(df, output_path)
    print("Preprocessing completed\n" + "=" * 50)


if __name__ == "__main__":
    main(opt["--input_file"], opt["--output_path"])
