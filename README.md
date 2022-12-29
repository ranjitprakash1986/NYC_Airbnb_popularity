# NYC Airbnb Popularity Predictor

  - author: Ranjitprakash Sundaramurthi
  - contributors: Andy Wang

A data analysis project in DSCI 573 (Data Science - Supervised Machine Learning); a
course in the Master of Data Science program at the University of
British Columbia. This project aims at discovering good Machine Learning models for predicting the popularity of an NYC Airbnb based on the provided dataset. 

## About

This significance of this project lies in the possibility that with a reliable Machine Learning model which can
estimate the popularity of an AirBnb listing, we can determine the parameters that influence popularity of
the Ad. This will in turn help the AirBnb hosts to write effective listings. It will help the company, AirBnb to
increase number of hosts that can meet these parameters. Thus AirBNB can revitalize its business model to
increase profits by focusing attention selective and promising listings. It will also enable the users/renters to
have a more positive experience in using AirBnb.

The Data for this project was sourced from Kaggle.com. Specifically it can be manually downloaded from
[here] (https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data). It can also be downloaded using the provided scripts in this repository.  
Each row in the data set represents the attributes of a specific ad listing in the NYC area. The specific attributes in the data set and there descriptions are mentioned below:

`name` : The title of the listing that the host has listed on AirBnB.
`host_id` and `host_name` : Id and name of the host for Airbnb listing.
`Neighbourhood_group` : The location in NYC where the listing is present. There are 5 unique values.
`neighbourhood` : The specific neighborhood withing the group where the property is location.
`latitude` and `longitude` : The geographical coordinates of the location.
`room_type` : The type of room the property. This can be entire home, private room or shared room.
`price` : The listed price per night on the Ad.
`minimum_nights` : The minimum number of nights the property can be booked for.
`number_of_reviews` : The number of user reviews previously posted on the property by users.
`last_review` : The date of the last review made on the property.
`reviews_per_month` : This is the proxy for the target that we need to predict. This represents the number
of reviews on average per month for the property.
`calculated_host_listings_count` : The total number of listings per host.
`availability_365` : The numebr of days in the year that property is available to occupy.

## Report

The final report can be found at
[here](**TBD**).

## Usage

There are two suggested ways to run this analysis:

#### 1\. Using Docker

*note - the instructions in this section also depends on running this in
a unix shell (e.g., terminal or Git Bash)*

To replicate the analysis, install
[Docker](https://www.docker.com/get-started). Then clone this GitHub
repository and run the following command at the command line/terminal
from the root directory of this project:

    docker run --rm -v /$(pwd):/home/rstudio/breast_cancer_predictor ttimbers/bc_predictor:v4.0 make -C /home/rstudio/breast_cancer_predictor all

To reset the repo to a clean state, with no intermediate or results
files, run the following command at the command line/terminal from the
root directory of this project:

    docker run --rm -v /$(pwd):/home/rstudio/breast_cancer_predictor ttimbers/bc_predictor:v4.0 make -C /home/rstudio/breast_cancer_predictor clean

#### 2\. Without using Docker

To replicate the analysis, clone this GitHub repository, install the
[dependencies](#dependencies) listed below, and run the following
command at the command line/terminal from the root directory of this
project:

    make all

To reset the repo to a clean state, with no intermediate or results
files, run the following command at the command line/terminal from the
root directory of this project:

    make clean

## Dependencies

  - Python 3.7.4 and Python packages:
      - docopt=0.6.2
      - requests=2.22.0
      - pandas=0.25.1R
      - feather-format=0.4.0
  - R version 3.6.1 and R packages:
      - knitr=1.26
      - feather=0.3.5
      - tidyverse=1.3.0
      - caret=6.0-85
      - ggridges=0.5.2
      - ggthemes=4.2.0
  - GNU make 4.2.1

## License

The NYC Airbnb Predictor materials here are licensed under the
Creative Commons Attribution (CC0 1.0 Universal (CC0 1.0) Public Domain Dedication). If
re-using/re-mixing please provide attribution and link to this webpage.

# References

<div id="refs" class="references hanging-indent">

<div id="ref-Dua2019">

Dua, Dheeru, and Casey Graff. 2017. “UCI Machine Learning Repository.”
University of California, Irvine, School of Information; Computer
Sciences. <http://archive.ics.uci.edu/ml>.

</div>

<div id="ref-Streetetal">

Street, W. Nick, W. H. Wolberg, and O. L. Mangasarian. 1993. “Nuclear
feature extraction for breast tumor diagnosis.” In *Biomedical Image
Processing and Biomedical Visualization*, edited by Raj S. Acharya and
Dmitry B. Goldgof, 1905:861–70. International Society for Optics;
Photonics; SPIE. <https://doi.org/10.1117/12.148698>.

</div>

</div>
