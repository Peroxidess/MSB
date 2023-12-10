# Codebase for "An active learning method based on multi-scale bin fuzz for selecting informative data in electronic health records"


This directory contains implementations of MSB framework for active learning.

Simply run python3 -m main.py

### Code explanation

(1) preprocess/load_data.py
- Load data

(2) preprocess/get_dataset.py
- Data preprocessing

(3) preprocess/missing_values_imputation.py
- Simple imputate missing values in dataset

(4) model/ae.py
- Define and return an autoencoder model to encode features

(5) model/evaluate.py
- Performance of computation in prediction tasks

(6) model/ActiveLearning.py
- Define MSB framework and model

(7) main.py
- A framework for data loading, model construction, result evaluation

(8) arguments.py
- Parameter settings

Note that hyper-parameters should be optimized for different datasets.


## Main Dependency Library
numpy==1.18.5

pandas==1.1.5

scikit-learn==0.24.2

torch==1.10.2

torchvision==0.11.3


