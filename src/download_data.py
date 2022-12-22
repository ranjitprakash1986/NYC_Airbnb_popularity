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
# python src/download_data.py --url="https://raw.githubusercontent.com/ranjitprakash1986/datasets/main/AB_NYC_2019.csv" --extract_to="data/raw/data.csv"
"""

# imports

import os
import pandas as pd
from docopt import docopt
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
    >>> main("https://raw.githubusercontent.com/ranjitprakash1986/datasets/main/AB_NYC_2019.csv" ,"data/raw/data.csv")
    """

    # Try catch to check if the url is valid
    try:
        print("Checking URL")
        request = requests.get(url)
        if request.status_code == 200:
            print("URL valid")
    except Exception as req:
        print(req)
        print("URL invalid")
    
    data = pd.read_csv(url, sep=',')

    # Save the file to the target path
    try:
        print("Saving CSV started")
        data.to_csv(extract_to, index=False)
        print("Save Completed")
    except:
        os.makedirs(os.path.dirname(extract_to))
        data.to_csv(extract_to, index=False)
        print("New folder created and save completed")

if __name__ == "__main__":
    main(opt["--url"], opt["--extract_to"])

    
    

