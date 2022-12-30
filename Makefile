# NYC Airbnb data pipe
# author: Ranjitprakash Sundaramurthi
# date: 2022-12-22

# ** please activate the conda environment before using the Make
# This is a makefile for the NYC Airbnb popularity prediction, there are 3 commands to run the make file:
# 1. `make all`: generate the HTML report and run all the required dependencies files / programs.
# 2. `make <target files>`: only run the specified target files.
# 3. `make clean`: clean all the generated files, images, and report files.

all: doc/NYC_Airbnb_popularity_predictor.html

# download data
data/raw/data.csv: src/download_data.py
	python src/download_data.py --url="https://raw.githubusercontent.com/ranjitprakash1986/datasets/main/AB_NYC_2019.csv" --extract_to="data/raw/data.csv"
	
# prepprocess data
data/processed/processed_data.csv: data/raw/data.csv src/preprocessing.py
	python src/preprocessing.py --input_file="data/raw/data.csv" --output_path="data/processed/"

# Feature Engineering - Sentiment Analysis	
data/processed/processed_sentiment_data.csv: data/processed/processed_data.csv src/sentiment_engineering.py
	python src/sentiment_engineering.py --input_file="data/processed/processed_data.csv" --output_path="data/processed/" --text_col="name"
	
# Split data and Exploratory Data Analysis
data/processed/train.csv data/processed/test.csv results/distribution_plot_reviews_per_month.png results/distribution_plot_reviews_per_month_neighbourhood_group.png results/distribution_plot_reviews_per_month_room_type.png results/histogram_reviews_per_month.png results/count_barplot_neighbourhood_group.png results/count_barplot_room_type.png results/correlation_plot.png results/text_count_name.png: data/processed/processed_sentiment_data.csv src/general_EDA.py
	python src/general_EDA.py --input_file="data/processed/processed_sentiment_data.csv" --output_data="data/processed/" --target_col="reviews_per_month"

# Analysis
results/allmodel_train_results.csv: data/processed/train.csv data/processed/test.csv src/analysis.py
	python src/analysis.py --train_file="data/processed/train.csv" --test_file="data/processed/test.csv" --output_dir="results/"

# Render html report
doc/NYC_Airbnb_popularity_predictor.html: doc/NYC_Airbnb_popularity_predictor.Rmd doc/NYC_Airbnb_refs.bib results/allmodel_train_results.csv results/best_parameters_lgbm.csv results/lgbmregressor_training_results.csv results/model_decision_tree  results/model_GradientBoosting results/model_lasso results/model_lgbm results/model_lgbmregressor results/model_linearregression results/model_random_forest results/model_ridgecv results/model_xgboost results/model_scores.csv
	Rscript -e "rmarkdown::render('doc/NYC_Airbnb_popularity_predictor.Rmd')"

clean: 
	rm -rf data
	rm -rf results
	rm -rf doc/NYC_Airbnb_popularity_predictor.html
			