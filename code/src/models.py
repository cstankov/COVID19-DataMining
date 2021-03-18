from main import *

pkl_ada_filename = "ada_pickle_model.pkl"
BASE_PATH = os.path.dirname(__file__)
BASE_DATA = '../models/'

def loadAda():
    # From https://stackabuse.com/scikit-learn-save-and-restore-models/
    file_pth = getRelPath(BASE_PATH, BASE_DATA, pkl_ada_filename)
    pickle_model = None
    with open(file_pth, 'rb') as file:
        pickle_model = pickle.load(file)

    return pickle_model    

def splitForModel(train_data, val_data, return_numpy = True):
    le = preprocessing.LabelEncoder()
    train_cpy = train_data.copy()
    val_cpy = val_data.copy()

    train_cpy = train_cpy.apply(le.fit_transform)
    val_cpy = val_cpy.apply(le.fit_transform)

    x_train = train_cpy.loc[:, train_cpy.columns != 'Outcome']
    y_train = train_cpy['Outcome']
    x_val = val_cpy.loc[:, val_cpy.columns != 'Outcome']
    y_val = val_cpy['Outcome']

    if(return_numpy == True):
        x_train = x_train.to_numpy()
        y_train = y_train.to_numpy()
        x_val = x_val.to_numpy()
        y_val = y_val.to_numpy()

    return x_train, y_train, x_val, y_val    

def runAdaBoostingClassifier(train_data, val_data):
    x_train, y_train, x_val, y_val = splitForModel(train_data, val_data)
    model = loadAda()
    



def adaBoostModel(train_data, val_data):
    # from https://www.datacamp.com/community/tutorials/adaboost-classifier-python
 
    x_train, y_train, x_val, y_val = splitForModel(train_data, val_data)
    ada = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(max_depth=2),
                            algorithm="SAMME.R", n_estimators = 200, learning_rate = 0.2)
    model = ada.fit(x_train, y_train)
 
    # From https://stackabuse.com/scikit-learn-save-and-restore-models/
    file_pth = getRelPath(BASE_PATH, BASE_DATA, pkl_ada_filename)
    with open(file_pth, 'wb') as file:
        pickle.dump(model, file)                

