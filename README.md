# Data Science portfolio_projects
This repository contains portfolio of data science projects resulted from my data science learning journey.
## Deep Learning- Computer vision projects
* [Mammogram breast-cancer detection]()
The goal of this competition is to identify breast cancer. The model was  trained with over 50000 screening mammograms obtained from regular screening.The goal of the  work is to help improve  the automation of detection in screening mammography and enable radiologists to be more accurate and efficient, improving the quality and safety of patient care. It could also help reduce costs and unnecessary medical procedures.
I developed a deep learning model for breast cancer prediction using TensorFlow and Keras frameworks, utilizing a pre-trained ResNet50 model and a custom-built convolutional neural network (CNN) for image classification.Converted over 54,000 DICOM training images (over 300 GB of data) into PNG format of different sizes (512x512, 256x256) using custom-built pipelines.Preprocessed the highly non-uniform DICOM scans in terms of patients, machine, exposure, size, etc. to prepare the data for training, using OpenCV and PIL libraries.Improved the accuracy and reduced runtime by implementing a transfer learning approach to train the ResNet50 model.

* [Colon cancer detection-histological_slides](https://github.com/lchudal89/portfolio_projects/blob/main/Computer_vision/colon_cancer%20prediction/coloncancer-training.ipynb)
Developed CNN model using 5000 histological images (for each cancer and normal) for colon cancer. Experiemnted with different transfer learning (mobileNet, ResNet, Xception) and finetuned the model. 
In this project, CNN  deeplearning framework keras and tensorflow were used to develop a image classification method  histological slides of colon cancer biopsy were 
* [NFL player contact's detection](https://github.com/lchudal89/portfolio_projects/blob/main/Computer_vision/NFL_players_contact_detection/Contact-detection-kaggle.ipynb)
Kaggle competitoion- In this kaggle competition, I developed a model to predict the external contact experienced by players during NFL football games, using video and player tracking data to improve player safety Final rank: top 18% (165/939)
## Machine learning
* [Hypothetical SpaceX launch prediction]()
* [Fraudelent credit card transaction prediction](https://github.com/lchudal89/portfolio_projects/blob/main/Machine_learning_projects/Credit-card%20fraud%20detection.ipynb)
Worked with highly imbalanced credit card transaction data to determine the fraudelent transaction. Experimented with various approaches to improve the accuracy of the models. Experimented with Decision tree classifier, random forest classifier, and XGBoosting to improve the f1- score.
* [Housing price prediction](https://github.com/lchudal89/portfolio_projects/blob/main/Machine_learning_projects/paris-housing-prediction.ipynb)
Experimented with linear regression, decision tree regressor, and random forest regressor to predict paris housing price. Used Grid-search to finetune the hyperparameters for decision tree and random forest regressor.
## Natural Language Processing
* [Movie_recommendation_system]()
Developed a movie recommendation system using contents (description, title, cast, creww,etc.) and collaborative filtering. 
* [curriculum recommendation system](https://github.com/lchudal89/portfolio_projects/blob/main/Natural_language_processing/curriculum-recommendation-system.ipynb)
The goal of this competition is to streamline the process of matching educational content to specific topics in a curriculum. I will develop an accurate and efficient model trained on a library of K-12 educational materials that have been organized into a variety of topic taxonomies. These materials are in diverse languages, and cover a wide range of topics, particularly in STEM (Science, Technology, Engineering, and Mathematics).
Developed a personalized curriculum recommendation system using NLP techniques for more than 50 languages, resulting in enhanced learning experiences and outcomes for students around the world.Implemented word preprocessing techniques such as tokenization, stop word removal, stemming, lemmatizing, and lowercasing to prepare content for analysis. Conducted experiments with various filtering approaches, such as language, creator, format, and hierarchy, to improve the accuracy (F2-score) and efficiency (run time) of the system. Utilized Python and NLP libraries (including NLTK and spaCy) to calculate cosine similarity between content topics.Ranked among the at top 16 out of over 1050 teams on the efficiency leaderboard.

## Time series analysis
* [US counties microbusiness density prediction](https://github.com/lchudal89/portfolio_projects/tree/main/Time_series_analysis)
Developed a time series forecasting model to predict microbusiness density in 3165 counties across the United States for the next 8 months, resulting in 25080 individual predictions. The model was trained on four years' worth of past data.Experimented with multiple algorithms, including linear regression, rolling average, autocorrelations, Fourier calendar, structural time series, theta-model, and Prophet, to identify the most effective features for improving the model's accuracy, such as trend, seasonality, and lag features. Implemented the identified features to improve the model's accuracy and successfully forecasted microbusiness density.Ranked top 30%.
[Linear regression](https://github.com/lchudal89/portfolio_projects/blob/main/Time_series_analysis/linear-regression-lag_features_train_test_split.ipynb) along with lag-features and seasonality, [Structural time series](https://github.com/lchudal89/portfolio_projects/blob/main/Time_series_analysis/GodaddyMBD_Structural_time_series.ipynb), and [theta model](https://github.com/lchudal89/portfolio_projects/blob/main/Time_series_analysis/GodaddyMBD_Theta%20model.ipynb) were used.
## Exploratory data analysis and visualization
* [ANPA members map](https://nbviewer.org/github/lchudal89/geomap/blob/main/anpa_members_map.ipynb)
In this project, I volunetereed to create location map of [ANPA-American physicsists] (https://anpaglobal.org/) members. I preprocessed the members data and  used Folium in python to create some beautiful maps.
* Exploratory data analysis and visualization of  [Housing price Kings county-USA](https://github.com/lchudal89/portfolio_projects/blob/main/Data%20analysis%20and%20visualizations/House_Sales_in_King_Count_USA.ipynb)
* Exploratory data analysis and visulization of [Canada immigration trend](https://github.com/lchudal89/portfolio_projects/blob/main/Data%20analysis%20and%20visualizations/Immigration_to_cananda.ipynb)
