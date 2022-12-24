# Author: Ranjit Sundaramurthi
# Date:  2022-12-21


"""
Script to split the data in train and test (20%)
Perform the preliminary EDA on Train data only
The plots are saved to the results/EDA folder
The data are saved to the data/processed/ folder

Usage: src/general_EDA.py --input_file=<input_path> --output_data=<output_path> --target_col=<target_column>

Options:
--input_file=<input_file>       Input file that has the processed data
--output_data=<output_path>     Path where the splitted data is to be stored
--target_col=<target_column>    Target column as a string

Example:
>>>python src/general_EDA.py --input_file="data/processed/processed_sentiment_data.csv" --output_data="data/processed/" --target_col="reviews_per_month"
"""

# imports
from docopt import docopt

import pandas as pd
import numpy as np

opt = docopt(__doc__)


def main(inp_file, out_data, tgt_col):
    """
    Main function to perform the genearl EDA

    Parameters:
    -----------
    inp_file:(str)          Input file with the processed data
    out_plot:(str)          Output path to save the plots
    out_data:(str)          Output path to save the data
    tgt_col:(str)           Target column

    Returns:
    ----------
    None
    """
    # imports
    from sklearn.model_selection import train_test_split
    
    # Read the processed data
    df = pd.read_csv(inp_file)
    
    # Split the data to train and test
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=573)

    # Storing the data
    try:
        train_df.to_csv(out_data + "train.csv", index=False)
        test_df.to_csv(out_data + "test.csv", index=False)
    except:
        os.makedirs(os.path.dirname(out_data))
        train_df.to_csv(out_data + "train.csv", index=False)
        test_df.to_csv(out_data + "test.csv", index=False)

    # Display message
    print("Data was split into 80% train and 20% test data in destination folder")
    print("="*50)
    
    print("Beginning generating EDA plots on training data only")
    print("="*50)

    # plot and save the target column density distribution
    col_distribution(train_df, tgt_col)
    
    # plot and save the target column histogram
    histogram(train_df, tgt_col, bins=50)
    
    # plot and save the categorical column barplots
    cat_count_plot(train_df, "neighbourhood_group")
    cat_count_plot(train_df, "room_type")
    
    # plot and save the correlation plot
    corr_plot(train_df)

    # plot and save the text count
    text_count(train_df, "name")
    
    # plot and save the target variable distribution plot by category room_type and neighborhood
    distribution_by_cat(train_df, 'reviews_per_month', 'room_type')
    distribution_by_cat(train_df, 'reviews_per_month', 'neighbourhood_group')
    
    # Display message
    print("EDA plots complete")
    print("="*50)
    
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
    None
    """
    
    # imports
    import altair as alt
    import os
    from utils.save_plot import save_chart
    
    alt.renderers.enable("mimetype")
    alt.data_transformers.enable("default", max_rows=None)
    
    # path
    dir_name = 'results'
    if os.path.isdir(dir_name)==False:
         os.makedirs(dir_name)
    filename = 'distribution_plot'+'_'+col_stringname
    extension = 'png'

    density_distribution_object = (
        alt.Chart(df)
        .transform_density(
            col_stringname, as_=[col_stringname, "density"], extent=[0, 10]
        )
        .mark_area()
        .encode(x=col_stringname, y="density:Q")
    )

    save_chart(density_distribution_object, os.path.join(dir_name, filename + '.' + extension))
    
    return


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
    None
    """

    # imports
    import matplotlib.pyplot as plt
    import os
    
    # path
    dir_name = 'results'
    if os.path.isdir(dir_name)==False:
         os.makedirs(dir_name)
    filename = 'histogram'+'_'+col_stringname
    extension = 'png'

    # target variable histogram distribution
    plt.clf()
    f, ax = plt.subplots()
    ax.hist(df[col_stringname], bins=50)
    f.savefig(os.path.join(dir_name, filename + '.' + extension))

    return


def cat_count_plot(df, cat_column_stringname):
    """
    Creates a barplot of the categorical column valuecounts

    Parameters:
    -----------
    df:                         Dataframe
    cat_column_stringname:      the column for which the chart is to be plotted

    Returns:
    ----------
    None
    """

    # imports
    import seaborn as sns
    import matplotlib.pyplot as plt
    import os
    
    # path
    dir_name = 'results'
    if os.path.isdir(dir_name)==False:
         os.makedirs(dir_name)
    filename = 'count_barplot'+'_'+cat_column_stringname
    extension = 'png'
    
    plt.clf()
    my_order = df[cat_column_stringname].value_counts().index
    ax = sns.countplot(x=df[cat_column_stringname], order=my_order)
    ax.set_title(cat_column_stringname)
    f = ax.get_figure()
    f.savefig(os.path.join(dir_name, filename + '.' + extension))
              
    return


def corr_plot(df):
    """
    Returns the correlation plot of the dataframe

    Parameters:
    -----------
    df:                         Dataframe

    Returns
    ----------
    None
    """

    # imports
    import numpy as np
    import dataframe_image as dfi
    import os
    
    # path
    dir_name = 'results'
    if os.path.isdir(dir_name)==False:
         os.makedirs(dir_name)
    filename = 'correlation_plot'
    extension = 'png'

    # numeric value correlation
    numeric_df = df.select_dtypes(include=np.number)
    
    # plot and save the correlation plot
    plot = numeric_df.corr().style.background_gradient()
    dfi.export(plot, os.path.join(dir_name, filename + '.' + extension))
    return


def text_count(df, col_stringname):
    """
    Returns a plot with frequency count of the text

    Parameter:
    ----------
    df:                         Dataframe
    col_stringname              Text Column with words

    Returns
    ---------
    None
    """

    # imports
    import nltk
    from nltk.corpus import stopwords
    from collections import Counter
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    
    # path
    dir_name = 'results'
    if os.path.isdir(dir_name)==False:
         os.makedirs(dir_name)
    filename = 'text_count'+'_'+col_stringname
    extension = 'png'

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
    ax.set_xticklabels(labels= data["Word"].values.tolist(), rotation=70)
    fig.savefig(os.path.join(dir_name, filename + '.' + extension))
    return

def distribution_by_cat(df, target_col, cat_col):
    """
    Returns a plot with target column distribution by category variable

    Parameter:
    ----------
    df:                     Dataframe
    target_col:             Target Column
    cat_col:                Category Column

    Returns
    ---------
    None
    """
    
    # imports
    import altair as alt
    import os
    from utils.save_plot import save_chart
    
    # path
    dir_name = 'results'
    if os.path.isdir(dir_name)==False:
         os.makedirs(dir_name)
    filename = 'distribution_plot'+'_'+target_col+'_'+cat_col
    extension = 'png'
    
    # target distribution for room type
    plot = alt.Chart(df).transform_density(
        target_col,
        as_=[target_col, 'density'],
        extent=[0, 10],
        groupby=[cat_col]
    ).mark_area(orient='horizontal').encode(
        y= alt.Y(target_col, type='quantitative'),
        color= alt.Color(cat_col, type='nominal'),
        x=alt.X(
            'density', type='quantitative',
            stack='center',
            impute=None,
            title=None,
            axis=alt.Axis(labels=False, values=[0],grid=False, ticks=True),
        ),
        column=alt.Column(
            cat_col, type='nominal',
            header=alt.Header(
                titleOrient='bottom',
                labelOrient='bottom',
                labelPadding=0,
            ),
        )
    ).properties(
        width=100
    ).configure_facet(
        spacing=0
    ).configure_view(
        stroke=None
    )
    
    save_chart(plot, os.path.join(dir_name, filename + '.' + extension))
    


if __name__ == "__main__":
    main(opt["--input_file"], opt["--output_data"], opt["--target_col"])
