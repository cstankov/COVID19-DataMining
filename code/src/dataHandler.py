from main import *

def preprocessing_data(location_data, test_data, main_train_data):
    BASE_PATH = os.path.dirname(__file__)
    BASE_DATA = '../results/'
    if os.path.exists(getRelPath(BASE_PATH, BASE_DATA,'location_transformed.csv')) and os.path.exists(getRelPath(BASE_PATH, BASE_DATA,'cases_test_processed.csv')) and os.path.exists(getRelPath(BASE_PATH, BASE_DATA,'cases_train_processed.csv')): 
        print("Preprocessed data Found!!")
        location_data, test_data_processed, train_data_processed = loadPreprocessedData()
        return location_data, test_data_processed, train_data_processed
    else:
        print("Preprocessing data...")
        #1.2
        data_cleaning_and_missing_values(location_data, test_data, main_train_data)
        print("Finished 1.2")
        #1.3
        find_outliers(main_train_data)
        print("Finished 1.3")
        #1.4
        location_data = transform_location_data(location_data)
        print("Finished 1.4")
        #1.5
        test_data_processed, train_data_processed = joining_datasets(location_data, test_data, main_train_data)
        print("Finished 1.5")
        saveData(location_data, test_data_processed, train_data_processed)
        #1.6
        print("Unique values in train data outcomes: ", np.unique(main_train_data['outcome']))
        return location_data, test_data_processed, train_data_processed

def split_train_val(train_data_processed):
    train_data, val_data = train_test_split(train_data_processed, test_size=0.2)
    return train_data, val_data

def loadPreprocessedData():
    BASE_PATH = os.path.dirname(__file__)
    BASE_DATA = '../results/'
    # na_lst = ['#N/A', '#N/A', 'N/A', '#NA', '-1.#IND', '-1.#QNAN', '-NaN', '-nan', '1.#IND', '1.#QNAN', '<NA>', 'N/A', 'NA', 'NULL', 'NaN', 'n/a', 'nan', 'null', 'Unknown', 'unknown']
    location_data = pd.read_csv(getRelPath(BASE_PATH, BASE_DATA,'location_transformed.csv'))
    test_data_processed = pd.read_csv(getRelPath(BASE_PATH, BASE_DATA, 'cases_test_processed.csv'))
    train_data_processed = pd.read_csv(getRelPath(BASE_PATH, BASE_DATA, 'cases_train_processed.csv'))
    return location_data, test_data_processed, train_data_processed

def loadAllData():
    BASE_PATH = os.path.dirname(__file__)
    BASE_DATA = '../data/'
    na_lst = ['#N/A', '#N/A', 'N/A', '#NA', '-1.#IND', '-1.#QNAN', '-NaN', '-nan', '1.#IND', '1.#QNAN', '<NA>', 'N/A', 'NA', 'NULL', 'NaN', 'n/a', 'nan', 'null', 'Unknown', 'unknown']
    location_data = pd.read_csv(getRelPath(BASE_PATH, BASE_DATA,'location.csv'), na_values=na_lst)
    test_data = pd.read_csv(getRelPath(BASE_PATH, BASE_DATA, 'cases_test.csv'), na_values=na_lst)
    main_train = pd.read_csv(getRelPath(BASE_PATH, BASE_DATA, 'cases_train.csv'), na_values=na_lst)
    return location_data, test_data, main_train

def saveData(location_data, test_data, train_data):
    BASE_PATH = os.path.dirname(__file__)
    BASE_DATA = '../results/'
    loc_path = getRelPath(BASE_PATH, BASE_DATA, 'location_transformed.csv')
    test_path = getRelPath(BASE_PATH, BASE_DATA, 'cases_test_processed.csv')
    train_path = getRelPath(BASE_PATH, BASE_DATA, 'cases_train_processed.csv')

    if os.path.exists(loc_path):
        os.remove(loc_path)

    if os.path.exists(train_path):
        os.remove(train_path)

    if os.path.exists(test_path):
        os.remove(test_path)

    location_data.to_csv(loc_path)
    test_data.to_csv(getRelPath(BASE_PATH, BASE_DATA, 'cases_test_processed.csv'), index=False)
    train_data.to_csv(getRelPath(BASE_PATH, BASE_DATA, 'cases_train_processed.csv'), index=False)

def getRelPath(BASE_PATH, BASE_DATA, SUB_PATH):
    COMBINED_PATH = BASE_DATA + SUB_PATH
    COMBINED_PATH = os.path.relpath(COMBINED_PATH, BASE_PATH)
    return COMBINED_PATH

def special_load():
    location_data = pd.read_csv('location_transformed.csv')
    test_data = pd.read_csv('cases_test_processed.csv')
    train_data = pd.read_csv('cases_train_processed.csv')
    return location_data, test_data, train_data