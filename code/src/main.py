import pandas as pd
import numpy as np
import os
import pickle

import matplotlib.pyplot as plt
import statistics
import hashlib

from IPython.display import display
from datetime import datetime

from scipy.stats import zscore
from sklearn.metrics import confusion_matrix
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import make_scorer, f1_score, recall_score, precision_recall_fscore_support, precision_score
from sklearn.preprocessing import OneHotEncoder
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
    x_train, y_train, x_val, y_val, test_data = splitForModel(test_data=test_data_processed, train_data = train_data, val_data = val_data)

    # # LIGHTGBM    
    # LGBMModelSave(x_train, y_train, x_val, y_val, test_data)
    # runLGBM(x_train, y_train, x_val, y_val, test_data)
    # overfitting_check(x_train, y_train, x_val, y_val, test_data)
    # runLGBM_hypertuned(x_train, y_train, x_val, y_val, test_data)
    # plot_result("LGBM", "../results/LGBM_tuning.csv", x_train, y_train, x_val, y_val, test_data)
    
    # # Linear SVC
    # linearSVMModelSave(x_train, y_train, x_val, y_val)
    # runLinearSVCClassifier(x_train, y_train, x_val, y_val)
    # overfittingLinearSVCcheck(x_train, y_train, x_val, y_val)
    # runLinearSVC_hypertuned(x_train, y_train, x_val, y_val)
    plot_result("LinearSVC", "../results/linearSVC_tuning.csv", x_train, y_train, x_val, y_val, test_data)
    
    #randomForests
    #The pkl model was too large for git buffers so I reduced the forest to only 1 tree
    #To change this, please go to models.py, line 275 and change the parameter
    # randomForestModelx_train, y_train, x_val, y_val, test_data)
    # runRandomForestClassifier(x_train, y_train, x_val, y_val, test_data)
    # random_forest_test_hparam(x_train, y_train, x_val, y_val, test_data)

    #For the best model
    # lgbm_predict(x_train, y_train, x_val, y_val, test_data)
    # print("checking...")
    # check_if_file_valid('../results/predictions.txt')   

if __name__ == "__main__":
    main()