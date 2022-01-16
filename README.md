# COVID19 - Data Mining

Since the first outbreak of COVID-19, millions of people around the world have been diagnosed with the illness. From this there have been many different agencies that have been collecting data on the disease throughout its lifespan. The three datasets that we analyzed are open source COVID-19 datasets that are operated by The John Hopkins University Center for Systems Science and Engineering. The information that is contained within these datasets is a wide set of attributes that are related to countries and people who are affected by the pandemic. The purpose of this study is to use the datasets to predict if a given patient has an outcome of being hospitalized, non-hospitalized, recovered or deceased. 

## Usage

Uncomment The models that you wish to be run in the main file located in the src directory. 
The main file starts out by preprocessing and split the datasets if the datasets have not been already. 
The Classifiers are then trained on the train data and run on the test data.
Hypertuning for the models is then conducted before plotting the results. 

### To Run:
`python3 main.py`
