# Author: Ranjit Sundaramurthi
# Date:  2022-12-21


"""
Script to perform the preliminary EDA
These are saved to the results/EDA folder

Usage: src/general_EDA.py --input_path=<input_path> --output_path=<output_path> --target_col=<target_column>

Options:
--input_path=<input_path>       Input path where the preprocessed data is present
--output_path=<output_path>     Path where the output plots are to be stored
--target_col=<target_column>    Target column as a string

Example:
>>>python src/general_EDA.py --input_path="data/processed/train_eda.csv" --output_path="results/EDA"
"""

# imports
from docopt import docopt

import pandas as pd
import numpy as np

opt = docopt(__doc__)


def main(inp_path, out_path, tgt_col):
    """
    Main function to perform the genearl EDA

    Parameters:
    -----------
    inp_path:(str)          Input path of the processed data
    out_path:(str)          Output path to save the plots
    tgt_col:(str)           Target column

    Returns:
    ----------
    None
    """

    # Read the processed data
    df = pd.read_csv(inp_path)

    # plot and save the target column density distribution
    col_distribution(df, tgt_col)
    # save plot

    # plot and save the target column histogram
    histogram(df, tgt_col, bins=50)
    # save plot

    # plot and save the categorical column barplots
    cat_count_plot(df, "neighbourhood_group")
    # save plot

    cat_count_plot(df, "room_type")
    # save plot

    # plot and save the correlation plot
    corr_plot(df)
    # save plot

    # plot and save the text count
    text_count(df, "name")
    # save plot

    return


def col_distribution(df, col_stringname):
    """
    Creates a density distribution of the column in df passed as a string

    Parameters
    ----------
    df:                           Dataframe
    col_stringname:               the column for which the density distribution is to be plotted

    Returns:
    ----------
    density_distribution: Chart object
    """
    import altair as alt

    alt.renderers.enable("mimetype")
    alt.data_transformers.enable("default", max_rows=None)

    density_distribution_object = (
        alt.Chart(df)
        .transform_density(
            col_stringname, as_=[col_stringname, "density"], extent=[0, 10]
        )
        .mark_area()
        .encode(x=col_stringname, y="density:Q")
    )

    return density_distribution_object


def histogram(df, col_stringname, bins=50):
    """
    Creates a histogram plot of the column passed from the dataframe

    Parameters
    ----------
    df:                         Dataframe
    col_stringname:             the column for which the density distribution is to be plotted
    bins:  bins for the histogram, default = 50

    Returns:
    ---------
    histogram_object:           Chart object
    """

    # imports
    import matplotlib.pyplot as plt

    # target variable histogram distribution
    histogram_object = plt.hist(df[col_stringname], bins=50)

    return histogram_object


def cat_count_plot(df, cat_column_stringname):
    """
    Creates a barplot of the categorical column valuecounts

    Parameters:
    -----------
    df:                         Dataframe
    cat_column_stringname:      the column for which the chart is to be plotted

    Returns:
    ----------
    ax:                         Plot object
    """

    # imports
    import seaborn as sns
    import matplotlib.pyplot as plt

    my_order = df[cat_column_stringname].value_counts().index
    ax = sns.countplot(x=df[cat_column_stringname], order=my_order)
    ax.set_title(cat_column_stringname)
    return ax


def corr_plot(df):
    """
    Returns the correlation plot of the dataframe

    Parameters:
    -----------
    df:                         Dataframe

    Returns
    ----------
    plot:                       Correlation plot object
    """

    # imports
    import numpy as np

    # numeric value correlation
    numeric_df = df.select_dtypes(include=np.number)

    plot = numeric_df.corr().style.background_gradient()
    return plot


def text_count(df, col_stringname):
    """
    Returns a plot with frequency count of the text

    Parameter:
    ----------
    df:                         Dataframe
    col_stringname              Text Column with words

    Returns
    ---------
    plot:                       Barplot object
    """

    # imports
    import nltk
    from nltk.corpus import stopwords
    from collections import Counter
    import matplotlib.pyplot as plt
    import seaborn as sns

    ## word count
    words = df[col_stringname].tolist()
    word_list = []
    for word in words:
        txt = str(word).split()
        for i in txt:
            word_list.append(i.lower())

    word_notin_eng = []
    for word in word_list:
        if word not in stopwords.words("english"):
            word_notin_eng.append(word)

    # count and get most 50 words
    count_result = Counter(word_notin_eng)
    most = count_result.most_common()[:50]
    data = pd.DataFrame(most, columns=["Word", "Count"])

    # plot
    fig = plt.figure(figsize=(15, 5))
    ax = sns.barplot(x="Word", y="Count", data=data)
    ax.xticks(rotation=70)
    return ax


if __name__ == "__main__":
    main(opt["--input_path"], opt["--output_path"], opt["--target_col"])
