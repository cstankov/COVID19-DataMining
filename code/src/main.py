import pandas as pd
import numpy as np
import os
import pickle

import matplotlib.pyplot as plt
import statistics

from datetime import datetime
from geopy.geocoders import Nominatim
from scipy.stats import zscore
from sklearn.metrics import confusion_matrix
from sklearn.metrics import multilabel_confusion_matrix

from sklearn.model_selection import train_test_split
from sklearn.ensemble import *
from sklearn import metrics
from sklearn import preprocessing
from sklearn.tree import *

from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.datasets import make_classification
from sklearn.datasets import make_multilabel_classification
from sklearn.multiclass import OneVsRestClassifier
from sklearn.calibration import CalibratedClassifierCV

from dataHandler import*
from helper1_2 import *
from helper1_3 import *
from helper1_4 import *
from helper1_5 import *
from models import *

def main():
    
    # location_data, test_data, main_train_data = loadAllData()
    # #1.2
    # data_cleaning_and_missing_values(location_data, test_data, main_train_data)
    # print("Finished 1.2")
    # #1.3
    # find_outliers(main_train_data)
    # print("Finished 1.3")
    # #1.4
    # location_data = transform_location_data(location_data)
    # print("Finished 1.4")
    # #1.5
    # test_data_processed, train_data_processed = joining_datasets(location_data, test_data, main_train_data)
    # print("Finished 1.5")
    # saveData(location_data, test_data_processed, train_data_processed)
    # #1.6
    # print("Unique values in train data outcomes: ", np.unique(main_train_data['outcome']))

    location_data, test_data_processed, train_data_processed = loadPreprocessedData()
    train_data, val_data = split_train_val(train_data_processed)

    # ADA
    adaBoostModelSave(train_data, val_data)
    runAdaBoostingClassifier(train_data, val_data)

    # Linear SVC
    # linearSVMModelSave(train_data, val_data)
    # runLinearSVCClassifier(train_data, val_data)



if __name__ == "__main__":
    main()