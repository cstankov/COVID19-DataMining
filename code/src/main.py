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
import lightgbm as lgbm

from sklearn.model_selection import train_test_split
from sklearn.ensemble import *
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.tree import *

from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.datasets import make_classification
from sklearn.datasets import make_multilabel_classification
from sklearn.multiclass import OneVsRestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score

from dataHandler import*
from helper1_2 import *
from helper1_3 import *
from helper1_4 import *
from helper1_5 import *
from models import *

def main():
    
    location_data, test_data, main_train_data = loadAllData()

    location_data, test_data_processed, train_data_processed  = preprocessing_data(location_data, test_data, main_train_data)
    train_data, val_data = split_train_val(train_data_processed)

    # LIGHTGBM
    # Run LightGBM 2.2 and 2.3
    LGBMModelSave(train_data, val_data)
    runLGBM(train_data, val_data)
    
    # # LGBM Overfitting Check 2.4
    # overfitting_check(train_data, val_data)

    # Linear SVC
    # linearSVMModelSave(train_data, val_data)
    # runLinearSVCClassifier(train_data, val_data)

    #randomForests
    #The pkl model was too large for git buffers so I reduced the forest to only 1 tree
    #To change this, please go to models.py, line 275 and change the parameter
    # randomForestModel(train_data, val_data)
    # runRandomForestClassifier(train_data, val_data)


if __name__ == "__main__":
    main()