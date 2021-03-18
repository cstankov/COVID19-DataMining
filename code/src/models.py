from main import *
from dataHandler import *

pkl_ada_filename = "ada_pickle_model.pkl"
BASE_PATH = os.path.dirname(__file__)
BASE_DATA = '../models/'

def loadAda():
    # From https://stackabuse.com/scikit-learn-save-and-restore-models/
    file_pth = os.path.relpath(BASE_DATA + pkl_ada_filename, BASE_PATH)
    pickle_model = None
    print("Loading AdaBoost Model")
    with open(file_pth, 'rb') as file:
        pickle_model = pickle.load(file)

    return pickle_model    

def splitForModel(train_data, val_data, return_numpy = True, convert_categorical = True):
    print("Splitting Model...")
    train_cpy = train_data.copy()
    val_cpy = val_data.copy()

    if convert_categorical == True:
        le = preprocessing.LabelEncoder()
        train_cpy = train_cpy.apply(le.fit_transform)
        val_cpy = val_cpy.apply(le.fit_transform)

    x_train = train_cpy.loc[:, train_cpy.columns != 'Outcome']
    y_train = train_cpy['Outcome']
    x_val = val_cpy.loc[:, val_cpy.columns != 'Outcome']
    y_val = val_cpy['Outcome']

    if return_numpy == True:
        x_train = x_train.to_numpy()
        y_train = y_train.to_numpy()
        x_val = x_val.to_numpy()
        y_val = y_val.to_numpy()

    return x_train, y_train, x_val, y_val    

def runAdaBoostingClassifier(train_data, val_data):
    x_train, y_train, x_val, y_val = splitForModel(train_data, val_data)
    model = loadAda()
    
    y_pred = model.predict(x_val)

    cf = confusion_matrix(y_val, y_pred)
    outputConfusionMatrixMetrics(cf)


def outputConfusionMatrixMetrics(confusion_mat):
    #From https://stackoverflow.com/questions/31324218/scikit-learn-how-to-obtain-true-positive-true-negative-false-positive-and-fal
    FP = confusion_mat.sum(axis=0) - np.diag(cf)  
    FN = confusion_mat.sum(axis=1) - np.diag(cf)
    TP = np.diag(confusion_mat)
    TN = confusion_mat.sum() - (FP + FN + TP)

    # Overall accuracy
    acc = (TP+TN)/(TP+FP+FN+TN)
    recall = TP /(TP + FN)
    prec = TP / (TP + FP)
    f_one_score = 2 / ( (1.0/recall) + (1.0/prec) )

    print("Accuracy: ", acc)
    print("Precision: ", prec)
    print("Recall: ", recall)
    print("F_One: ", f_one_score)



def adaBoostModelSave(train_data, val_data):
    # from https://www.datacamp.com/community/tutorials/adaboost-classifier-python
 
    x_train, y_train, x_val, y_val = splitForModel(train_data, val_data)

    print("Running Ada...")
    ada = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(max_depth=2),
                            algorithm="SAMME.R", n_estimators = 200, learning_rate = 0.2)
    model = ada.fit(x_train, y_train)
 
    print("Saving AdaBoost Model...")
    # From https://stackabuse.com/scikit-learn-save-and-restore-models/
    file_pth = os.path.relpath(BASE_DATA + pkl_ada_filename, BASE_PATH)
    with open(file_pth, 'wb') as file:
        pickle.dump(model, file)          


