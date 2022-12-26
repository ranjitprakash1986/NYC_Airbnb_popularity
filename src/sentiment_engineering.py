# Author: Ranjit Sundaramurthi
# Date: 2022-12-23

"""
Script to perform sentiment analysis engineering and add as feature

Usage: src/sentiment_engineering.py --input_file=<input_file> --output_path=<output_path> --text_col=<text_column>

Options:
--input_file=<input_file>       Input filename with path
--output_path=<output_path>     Output path where the datafrane is to be written
--text_col=<text_column>        Text column to be sentiment analyzed

Example:
>>>python src/sentiment_engineering.py --input_file="data/processed/processed_data.csv" --output_path="data/processed/" --text_col="name"

"""
# imports
from docopt import docopt
opt = docopt(__doc__)

def get_sentiment(text):
    """
    Returns the compound score representing the sentiment: -1 (most extreme negative) and +1 (most extreme positive)
    The compound score is a normalized score calculated by summing the valence scores of each word in the lexicon.

    Parameters:
    ------
    text: (str)
    the input text

    Returns:
    -------
    sentiment of the text: (str)
    """
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer

    sid = SentimentIntensityAnalyzer()

    scores = sid.polarity_scores(text)
    return scores["compound"]


def main(inp_file, output_path, text_col):
    """
    Adds features to provided dataframe and saves it

    Parameters:
    ----------
    df:         Dataframe

    Returns:
    ----------
    None
    """
    # imports
    import pandas as pd
    
    # Read the file
    df = pd.read_csv(inp_file)

    # Perform the sentiment analysis
    print("Starting sentiment analysis of the data")
    df = df.assign(vader_sentiment=df[text_col].apply(get_sentiment))

    # Storing the enhanced data
    try:
        df.to_csv(output_path + "processed_sentiment_data.csv", index=False)
    except:
        os.makedirs(os.path.dirname(output_path))
        df.to_csv(output_path + "processed_sentiment_data.csv", index=False)

    # Display message
    print("Sentiment feature was added and saved to destination folder")

if __name__ == "__main__":
    main(opt["--input_file"], opt["--output_path"], opt["--text_col"])