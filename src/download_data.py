# Author: Ranjitprakash Sundaramurthi
# Date: 2022-12-21

"""
Script to download data from specified URL.
Unzip it to the location specified by the user as save as data.csv

Usage: src/download_data.py --url=<url> --extract_to=<extract_to>

Options:
--url=<url>                 URL from where to download the data from
--extract_to=<extract_to>   The path where the file is to be unzipped to in the repository.

Example:
# python download_data.py --url="https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-datahttps://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data/download?datasetVersionNumber=3" --extract_to="../data/raw/data.csv"
"""

# imports

import os
import pandas as pd
import docopt as docopt
import requests
import urllib
import zipfile

opt = docopt(__doc__) #parse the docstring to a dictionary

def main(url, extract_to):
    """
    Download data from the url and extract to given location

    Parameters:
    ----------
    url: (str) The weblink from where to download the data
    extract_to: (str) The path where to extract the downloaded zip file

    Returns:
    --------
    Nothing

    Example:
    -------
    >>> main("https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-datahttps://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data/download?datasetVersionNumber=3" ,"../data/raw/data.csv")
    """

    # Try catch to check if the url is valid
    try:
        print("Checing URL")
        request = requests.get(url)
        if request.status_code == 200:
            print("URL valid")
    except Exception as req:
        print(req)
        print("URL invalid")
    

    # Try catch to downloada and extract the file
    try:
        zip_path, _ = urllib.request.urlretrieve(url)
        with zipfile.ZipFile(zip_path, "r") as f:
            f.extractall(extract_to)
        print("File downloaded and extracted successfull")
    except:
        os.makedirs(os.path.dirname(extract_to))
        with zipfile.ZipFile(zip_path, "r") as f:
            f.extractall(extract_to)
        print("File downloaded and extracted successfull")

if __name__ == "__main__":
    main(opt["--url"], opt["--extract_to"])

    
    

