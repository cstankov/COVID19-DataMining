from main import *
from dataHandler import *

pkl_ada_filename = "ada_pickle_model.pkl"
pkl_linearSVC_filename = "linearSVC_pickle_model.pkl"
BASE_PATH = os.path.dirname(__file__)
BASE_DATA = '../models/'   

def splitForModel(train_data, val_data, return_numpy = True, convert_categorical = True):
    print("Splitting Model...")
    train_cpy = train_data.copy()
    val_cpy = val_data.copy()

    if convert_categorical == True:
        le = preprocessing.LabelEncoder()
        train_cpy = train_cpy.apply(le.fit_transform)
        val_cpy = val_cpy.apply(le.fit_transform)
        # le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        # print("Mapping:")
        # print(le_name_mapping)
        # print("\n\n")
        
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


def printMetrics(confusion_mat, label_number):
    TN, FP, FN, TP = confusion_mat.ravel()

    if label_number == 0:
        print("Deceased:")
    elif label_number == 1:
        print("Hospitalized:")
    elif label_number == 2:
        print("Nonhospitalized:")
    elif label_number == 3:
        print("Recovered:")
    else:
        print("Invalid label number. Enter number between 0 and 3. Refer to the map in code")

    acc = (TP+TN)/(TP+FP+FN+TN)
    recall = TP /(TP + FN)
    prec = TP / (TP + FP)
    f_one_score = 2 / ( (1.0/recall) + (1.0/prec) )

    print("Accuracy: ", acc)
    print("Precision: ", prec)
    print("Recall: ", recall)
    print("F_One: ", f_one_score)       
    print("\n\n")     



def outputConfusionMatrixMetrics(confusion_mat, unique_labels):
    #From https://stackoverflow.com/questions/31324218/scikit-learn-how-to-obtain-true-positive-true-negative-false-positive-and-fal
    # print(unique_labels)
    print("CF:\n", confusion_mat)
    print("\n\n")

    '''
    0 --> deceased
    1 --> hospit..
    2 --> nonhosp...
    3 --> recovered

    Confusion Mat:
    TP  FN
    FP  TN
    '''
    
    cf_dec = confusion_mat[0]
    cf_hosp = confusion_mat[1]
    cf_nonhosp = confusion_mat[2]
    cf_rec = confusion_mat[3]

    printMetrics(cf_dec, 0)
    printMetrics(cf_hosp, 1)
    printMetrics(cf_nonhosp, 2)
    printMetrics(cf_rec, 3)


def loadModel(ModelFileName):
    # From https://stackabuse.com/scikit-learn-save-and-restore-models/
    file_pth = os.path.relpath(BASE_DATA + ModelFileName, BASE_PATH)
    pickle_model = None

    if ModelFileName == pkl_ada_filename:
        print("Loading Ada Model...")
    elif ModelFileName == pkl_linearSVC_filename:
        print("Loading SVC Model...")

    with open(file_pth, 'rb') as file:
        pickle_model = pickle.load(file)
    return pickle_model 

################ ADA BOOST MODEL ################
def overfitting_check(train_data, val_data):
    print("Checking for Overfitting...")
    #From https://machinelearningmastery.com/overfitting-machine-learning-models/
    x_train, y_train, x_val, y_val = splitForModel(train_data, val_data)
    train_scores, test_scores = list(), list()

    depth = [None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    for d in depth:
        DTC = DecisionTreeClassifier(random_state = 0, max_features = "auto", class_weight = "balanced",max_depth = d)
        # param_grid = {"base_estimator__criterion" : ["entropy"],
        #         "base_estimator__splitter" :   ["best"],
        #         "n_estimators": [200],
        #         "learning_rate": [1] 
        #         } 
        ada = AdaBoostClassifier(base_estimator = DTC)
        # grid_search_ada = GridSearchCV(ada, param_grid=param_grid)
        model = ada.fit(x_train, y_train)
        # evaluate on the train dataset
        train_y_pred = model.predict(x_train)
        train_acc = accuracy_score(y_train, train_y_pred)
        train_scores.append(train_acc)
        # evaluate on the test dataset
        test_y_pred = model.predict(x_val)
        test_acc = accuracy_score(y_val, test_y_pred)
        test_scores.append(test_acc)
        # summarize progress
        if d == None:
            print('>Max Depth: %s, train: %.3f, test: %.3f' % ('None', train_acc, test_acc))
        else:
            print('>Max Depth: %d, train: %.3f, test: %.3f' % (d, train_acc, test_acc))
    
    plt.plot(depth, train_scores, '-o', label='Train')
    plt.plot(depth, test_scores, '-o', label='Test')
    plt.title("AdaBoost Overfitting")
    plt.xlabel("Maximum Depth of Tree")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()



def adaBoostModelSave(train_data, val_data):
    # from https://www.datacamp.com/community/tutorials/adaboost-classifier-python
 
    x_train, y_train, x_val, y_val = splitForModel(train_data, val_data)
    print("Running Ada...")

    # Milestone3 
    # Used GridSearchCV to get best parameters and then made the best_parameter_grid
    # uncomment to run the posible parameter 
    # idea from https://stackoverflow.com/questions/32210569/using-gridsearchcv-with-adaboost-and-decisiontreeclassifier
    # DTC = DecisionTreeClassifier(random_state = 0, max_features = "auto", class_weight = "balanced",max_depth = None)
    # param_grid = {"base_estimator__criterion" : ["gini", "entropy"],
    #           "base_estimator__splitter" :   ["best", "random"],
    #           "n_estimators": [50, 100, 150, 200],
    #           "learning_rate": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1] #best --> not this
    #          } 
    # ada = AdaBoostClassifier(base_estimator = DTC)
    # grid_search_ada = GridSearchCV(ada, param_grid=param_grid)
    # model = grid_search_ada.fit(x_train, y_train)
    # print("Best Parameters: ", grid_search_ada.best_params_)
 
    # using best parameters
    # DTC = DecisionTreeClassifier(random_state = 0, max_features = "auto", class_weight = "balanced",max_depth = 5)
    # param_grid = {"base_estimator__criterion" : ["entropy"],
    #           "base_estimator__splitter" :   ["best"],
    #           "n_estimators": [200],
    #           "learning_rate": [1] 
    #          } 
    # ada = AdaBoostClassifier(base_estimator = DTC)
    # grid_search_ada = GridSearchCV(ada, param_grid=param_grid)
    # model = grid_search_ada.fit(x_train, y_train)


    # Milestone2:
    DTC = DecisionTreeClassifier()
    ada = AdaBoostClassifier(base_estimator = DTC)
    model = ada.fit(x_train, y_train)
    print("Saving AdaBoost Model...")
    # From https://stackabuse.com/scikit-learn-save-and-restore-models/
    file_pth = os.path.relpath(BASE_DATA + pkl_ada_filename, BASE_PATH)
    with open(file_pth, 'wb') as file:
        pickle.dump(model, file)          

def runAdaBoostingClassifier(train_data, val_data):
    x_train, y_train, x_val, y_val = splitForModel(train_data, val_data)
    model = loadModel(pkl_ada_filename)
    
    y_pred = model.predict(x_val)
    print("y_pred unique:", np.unique(y_pred))

    unique_labels = np.unique(y_val)

    cf = multilabel_confusion_matrix(y_val, y_pred, labels = unique_labels)
    outputConfusionMatrixMetrics(cf, unique_labels)

################ LINEAR SVC MODEL ################

def linearSVMModelSave(train_data, val_data):
    # from https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html
    # from https://stackoverflow.com/questions/31617530/multiclass-linear-svm-in-python-that-return-probability
    x_train, y_train, x_val, y_val = splitForModel(train_data, val_data)
    
    print("Running linear SVC...")
    cl = CalibratedClassifierCV(LinearSVC(C=1, dual=False, max_iter=10000))
    clf = OneVsRestClassifier(cl)
    model = clf.fit(x_train, y_train)

    print("Saving linear SVC Model...")
    # From https://stackabuse.com/scikit-learn-save-and-restore-models/
    file_pth = os.path.relpath(BASE_DATA + pkl_linearSVC_filename, BASE_PATH)
    with open(file_pth, 'wb') as file:
        pickle.dump(model, file)  

def runLinearSVCClassifier(train_data, val_data):
    x_train, y_train, x_val, y_val = splitForModel(train_data, val_data)
    model = loadModel(pkl_ada_filename)

    y_pred = model.predict(x_val)

    cf = confusion_matrix(y_val, y_pred)
    outputConfusionMatrixMetrics(cf)
