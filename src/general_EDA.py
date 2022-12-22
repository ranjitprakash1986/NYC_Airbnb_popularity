# Author: Ranjit Sundaramurthi
# Date:  2022-12-21


"""
Script to perform the preliminary EDA
These are saved to the results/EDA folder

Usage: src/general_EDA.py --input_path=<input_path> --output_path=<output_path>

Options:
--input_path=<input_path>   Input path where the preprocessed data is present
--output_path=<output_path> Path where the output plots are to be stored

Example:
>>>python src/general_EDA.py --input_path="data/processed/train_eda.csv" --output_path="results/EDA"
"""

# imports
from docopt import docopt

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt

# alt.renderers.enable('mimetype')
alt.data_transformers.enable("default", max_rows=None)

opt = docopt(__doc__)

# Read the dataframe


def distribution(df, col_stringname):
    """
    Creates a density distribution of the column in df passed as a string

    Parameters
    ----------

    Returns
    ---------
    """
    import altair as alt

    # alt.renderers.enable('mimetype')
    alt.data_transformers.enable("default", max_rows=None)

    alt.Chart(df).transform_density(
        col_stringname, as_=[col_stringname, "density"], extent=[0, 10]
    ).mark_area().encode(x=col_stringname, y="density:Q")


# Read the processed data
df = pd.read_csv(inp_path)

# drop rows with the missing values
house_df = house_df[house_df["name"].notna()]
house_df = house_df[house_df["host_name"].notna()]
house_df = house_df[house_df["last_review"].notna()]
house_df = house_df[house_df["reviews_per_month"].notna()]

# change dates of review to pandas inbuilt date
house_df["last_review"] = pd.to_datetime(
    house_df["last_review"], infer_datetime_format=True
)

# Create a new feature column "days_since_review"
house_df = house_df.assign(
    days_since_review=house_df["last_review"].apply(
        lambda x: (np.datetime64("today") - x).days
    )
)

# Now we can drop the last_review column
# as we have captured its effect through days_since_review
# Dropping host_id since it has unique identifier for most rows
house_df = house_df.drop(columns=["last_review", "host_id", "id"])

# Suggestion for feature transformation to reduce the number of categories
# to reduce the number of unique categories for host_name we could bunch together under "other"
name_df = pd.DataFrame(house_df["host_name"].value_counts())

# Determining the threshold for which roughly half of the data set is "other"
threshold = 14  # this moves fairly 50% to other category


name_other = name_df[name_df["host_name"] < threshold].index.tolist()
house_df = house_df.assign(
    host_name_mod=house_df["host_name"].apply(
        lambda x: "other" if x in name_other else x
    )
)
