This project includes our fake news detection model and a news scraping module, built using the Flask framework.

## Overview
The most important files in this projects are as follow:
- dataset: This folder contains the training and testing datasets.
- model: This folder includes the model architecture (layer.py), training (train.py) and evaluation (eval.py) scripts, as well as the dataset class (dataloader.py).
- search: This folder contains scripts for scrping news from four different websites (fenghuang, sina, souhu, wangyi).
- app.py: The main run file of this project.
- data_processing.py: Script for data preprocessing and preparation.
- object.py: Contains the definitions of key objects or classes used in the project.
- result.py: Responsible for encapsulating and formatting the output results returned by the project.
- services.py: Implements auxiliary services and business logic for the application.
- utils.py: Utility functions used throughout the project.
- requirements.py: Configuration file listing the required Python packages for this project.
- README.md: Readme file providing an overview and instructions for the project.


## Getting Started
If you want to quickly get started with this project, please refer to the following instructions.

### Installation
```
conda create -n main python=3.10
conda activate main
pip install -r requirements.txt
```

### Running
Before running the following commands, you must use your own API in the utils.py for the APIs.
```
cd fakeNews
python app.py
```


## Model
If you are instered in our model, please refer to the following steps.

### Data Preparation
```
python data_processing.py
```

### Training
```
cd fakeNews/model
python train.py
```
### Evaluation
```
python eval.py
```
