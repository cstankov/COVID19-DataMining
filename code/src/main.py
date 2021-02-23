import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import statistics
from datetime import datetime
from geopy.geocoders import Nominatim
from scipy.stats import zscore

from dataHandler import*
from helper1_2 import *
from helper1_3 import *
from helper1_4 import *
from helper1_5 import *

def main():
    location_data, test_data, train_data = loadAllData()
    #1.2
    data_cleaning_and_missing_values(location_data, test_data, train_data)
    print("Finished 1.2")
    #1.3
    # find_outliers(location_data, test_data, train_data)
    print("Finished 1.3")
    #1.4
    location_data = transform_location_data(location_data)
    print("Finished 1.4")
    #1.5
    test_data_processed, train_data_processed = joining_datasets(location_data, test_data, train_data)
    print("Finished 1.5")
    saveData(location_data, test_data_processed, train_data_processed)
    #1.6
    print("Unique values in train data outcomes: ", np.unique(train_data['outcome']))


if __name__ == "__main__":
    main()