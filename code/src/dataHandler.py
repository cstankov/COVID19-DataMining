from main import *

def loadAllData():
    BASE_PATH = os.path.dirname(__file__)
    BASE_DATA = '../data/'
    na_lst = ['#N/A', '#N/A', 'N/A', '#NA', '-1.#IND', '-1.#QNAN', '-NaN', '-nan', '1.#IND', '1.#QNAN', '<NA>', 'N/A', 'NA', 'NULL', 'NaN', 'n/a', 'nan', 'null', 'Unknown', 'unknown']
    location_data = pd.read_csv(getRelPath(BASE_PATH, BASE_DATA,'location.csv'), na_values=na_lst)
    test_data = pd.read_csv(getRelPath(BASE_PATH, BASE_DATA, 'cases_test.csv'), na_values=na_lst)
    train_data = pd.read_csv(getRelPath(BASE_PATH, BASE_DATA, 'cases_train.csv'), na_values=na_lst)
    return location_data, test_data, train_data

def saveData(location_data, test_data, train_data):
    BASE_PATH = os.path.dirname(__file__)
    BASE_DATA = '../results/'
    location_data.to_csv(getRelPath(BASE_PATH, BASE_DATA, 'location_transformed.csv'))
    test_data.to_csv(getRelPath(BASE_PATH, BASE_DATA, 'cases_test_processed.csv'))
    train_data.to_csv(getRelPath(BASE_PATH, BASE_DATA, 'cases_train_processed.csv'))

def getRelPath(BASE_PATH, BASE_DATA, SUB_PATH):
    COMBINED_PATH = BASE_DATA + SUB_PATH
    COMBINED_PATH = os.path.relpath(COMBINED_PATH, BASE_PATH)
    return COMBINED_PATH

def special_load():
    location_data = pd.read_csv('location_transformed.csv')
    test_data = pd.read_csv('cases_test_processed.csv')
    train_data = pd.read_csv('cases_train_processed.csv')
    return location_data, test_data, train_data