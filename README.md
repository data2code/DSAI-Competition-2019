# EC2 Team Solution in Novartis Data Science and Artificial Intelligence (DSAI) Challenge 2019


The [Novartis DSAI Challenge 2019](https://www.aicrowd.com/challenges/novartis-dsai-challenge) aimed at predicting the Probability of Success (PoS) of a clinical candidate for a certain indication based on the data collected on Phase II trials.  This repository contains the solution submitted by team E2C, consisting of members from the [Genomics Insitute of the Novartis Research Foundation](https://www.gnf.org/).

## Dataset

The training data was based on two proprietary pharmaceutical pipeline databases provided by [Informa](https://pharmaintelligence.informa.com/)&copy; ([Pharmaprojects](https://pharmaintelligence.informa.com/products-and-services/data-and-analysis/pharmaprojects) and [Trialtrove](https://pharmaintelligence.informa.com/clinical-trial-data)), and we therefore do not  provide the raw or processed training and inference data in this repository.

## Model Development Workflow

As mentioned, the code here does not run as it is, as it depends on both training datasets as well as other settings provided by the competition model evaluation docker environment.  Nevertheless, the syntax can help you understand the entry points in the main analysis code EDA_v4.py.

### Setting
[NLTK](https://pypi.org/project/nltk/) should be preinstalled.
```
export PYTHONPATH=./lib:$PYTHONPATH
export NLTK_DATA="./nltk_data"
```
### Usage
```
usage: EDA_v4.py [-h] [-p] [-e] [-i] [-r] [-d] [-s] [-k]

Phase2-Approval Prediction

optional arguments:
  -h, --help            show this help message and exit
  -p, --hyperparameter  hyperparameter tuning
  -e, --estimator       estimator tuning
  -r, --recreate        recreate feature matrix
  -s, --sort            sort features
```
### Example Workflow

#### Configure model with
```
configjson.py
```
#### To recreate feature matrix 
```
EDA_v4.py -r
```
This creates feature_matrix.pkl.gz

#### Hyperparameter tuning

Define hyperparameter search grid in params/hyperparameter.range.json
```
EDA_v4.py -p
```
This creates params/hyperparameter.best.json and and params/hyperparameter.estimator.json

Edit params/hyperparameter.range.json and run
```
EDA_v4.py -p
```
to refine number of estimator counts and learning rate
You may repeat this tuning process.

Final hyperparameters in params/hyperparameter.best.json
```
EDA_v4.py –e
```
This trains a model using all training data based on params/hyperparameter.best.json

The model file is model.pkl.gz

#### Model application
```
EDA_v4.py
```
## Additional Information

You can find more about DSAI 2019 and another winning model by Team [Insight-Out](https://github.com/bjoernholzhauer/DSAI-Competition-2019). 

## E2C Team Members

* Yang Zhong (yang.zhong at novartis dot com)
* Bin Zhou (bin.zhou at novartis dot com)
* Shifeng Pan (span at gnf dot org)
* Yingyao Zhou (yingyao.zhou at novartis dot com)
