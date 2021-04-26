from main import *
from dataHandler import *

pkl_lgbm_filename = "lgbm_pickle_model.pkl"
pkl_linearSVC_filename = "linearSVC_pickle_model.pkl"
pkl_ranforest_filename = "ranforest_pickle_model.pkl"
BASE_PATH = os.path.dirname(__file__)
BASE_DATA = '../models/'   
    
def splitForModel(test_data, train_data, val_data, return_numpy = True, convert_categorical = True):
    print("Splitting Model...")
    train_cpy = train_data.copy()
    val_cpy = val_data.copy()
    test_cpy = test_data.copy()

    train_cpy.drop(['Source'], axis = 1, inplace = True)
    val_cpy.drop(['Source'], axis = 1, inplace = True)
    test_cpy.drop(['Source'], axis = 1, inplace = True)
    
    print("Doing for OUTCOME")
    train_cpy['Outcome'].replace({'deceased':0, 'hospitalized':1, 'nonhospitalized':2, 'recovered':3},inplace=True)
    val_cpy['Outcome'].replace({'deceased':0, 'hospitalized':1, 'nonhospitalized':2, 'recovered':3},inplace=True)
    categorical_cols = ["Sex",  "Province_State", "Country", "Combined_Key", "Date_Confirmation"]

    if convert_categorical == True:
        for col in categorical_cols:
            train_cpy[col] = train_cpy[col].apply(lambda x: int(hashlib.sha1(x.encode("utf-8")).hexdigest(), 16)) % (10 ** 8)
            val_cpy[col] = val_cpy[col].apply(lambda x: int(hashlib.sha1(x.encode("utf-8")).hexdigest(), 16)) % (10 ** 8)
            test_cpy[col] = test_cpy[col].apply(lambda x: int(hashlib.sha1(x.encode("utf-8")).hexdigest(), 16)) % (10 ** 8)

    print("Done categorical encoding...")  
    x_train = train_cpy.loc[:, train_cpy.columns != 'Outcome']
    # test_cpy = test_cpy.loc[:, test_data.columns != 'Outcome']
    test_cpy.drop(['Outcome'], axis='columns', inplace=True)

    y_train = train_cpy['Outcome']
    x_val = val_cpy.loc[:, val_cpy.columns != 'Outcome']
    y_val = val_cpy['Outcome']

    if return_numpy == True:
        x_train = x_train.to_numpy()
        y_train = y_train.to_numpy()
        x_val = x_val.to_numpy()
        y_val = y_val.to_numpy()
        test_cpy = test_cpy.to_numpy()

    return x_train, y_train, x_val, y_val, test_cpy

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

    if ModelFileName == pkl_lgbm_filename:
        print("Loading LGBM Model...")
    elif ModelFileName == pkl_linearSVC_filename:
        print("Loading SVC Model...")

    with open(file_pth, 'rb') as file:
        pickle_model = pickle.load(file)
    return pickle_model 

def plot_result(model_name, model_result_path, x_train, y_train, x_val, y_val, test_data):
    if (model_name == 'LGBM'):
        if not (os.path.exists(model_result_path)):
            runLGBM_hypertuned(x_train, y_train, x_val, y_val, test_data)
        lgbm_result = pd.read_csv(model_result_path)
        tuned = list(zip(lgbm_result.param_num_leaves, lgbm_result.param_n_estimators, lgbm_result.param_max_bin))
        tuned_params = ['{}, {}, {}'.format(x,y,z) for x,y,z in tuned]
        params = "(num_leaves, n_etimators, max_bin)"
        plotFigures(lgbm_result, tuned_params, params, model_name)

    elif model_name == 'LinearSVC':
        if not (os.path.exists(model_result_path)):
            runLinearSVC_hypertuned(test_data, train_data = train_data, val_data = val_data)
        linearSVC_result = pd.read_csv(model_result_path)
        tuned = list(zip(linearSVC_result.param_estimator__C, linearSVC_result.param_estimator__class_weight, linearSVC_result.param_estimator__tol))
        tuned_params = ['{}, {}, {}'.format(x,y,z) for x,y,z in tuned]
        params = "(C, n_etimators, tol)"
        plotFigures(linearSVC_result, tuned_params, params, model_name)
        
    else: 
        print("Invalid Model name given. Expected 'LGBM', 'LinearSVC' or 'RandomForest', got: ", model_name)
 
def plotFigures(results, tuned_params, params, model):
    mean_test_accuracy = list(results['mean_test_Accuracy'])
    mean_train_accuracy = list(results['mean_train_Accuracy'])
 
    mean_test_f1_deceased = list(results['mean_test_f1_score_decease'])
    mean_train_f1_deceased = list(results['mean_train_f1_score_decease'])
 
    mean_test_recall = list(results['mean_test_recall'])
    mean_train_recall = list(results['mean_train_recall'])
 
    mean_test_recall_deceased = list(results['mean_test_recall_Deceased'])
    mean_train_recall_deceased = list(results['mean_train_recall_Deceased'])
 
    plt.figure() #accuracy
    plt.plot(tuned_params, mean_test_accuracy, label = "Test Accuracy", marker = 'o')
    plt.plot(tuned_params, mean_train_accuracy, label = "Train Accuracy", marker = 'o')
 
    plt.xlabel(params)
    plt.ylabel("Accuracy")
    model_title = model + " Model Accuracy vs tuned paramters"
    plt.title(model_title)
 
    plt.xticks(rotation=90)   
    plt.tight_layout()     
    plt.legend()
    accuracy_save = '../plots/' + model + '_accuracy_plot.png'
    plt.savefig(accuracy_save)
 
    plt.figure() #f1-score-deceased
    plt.plot(tuned_params, mean_test_f1_deceased, label = "Test F1 Score for Deceased", marker = 'o')
    plt.plot(tuned_params, mean_train_f1_deceased, label = "Train F1 Score for Deceased", marker = 'o')
 
    plt.xlabel(params)
    plt.ylabel("F1 SCore for 'Deceased'")
    model_title = model + "  Model F1 Score for 'Deaceased' Outcome vs tuned paramters"
    plt.title(model_title)
 
    plt.xticks(rotation=90)        
    plt.tight_layout()     
    plt.legend()
    f1_score_deceased_save = '../plots/' + model + '_F1_score_deceased_plot.png'
    plt.savefig(f1_score_deceased_save)
 
    plt.figure() #recall-deceased
    plt.plot(tuned_params, mean_test_recall_deceased, label = "Test Recall for Deceased", marker = 'o')
    plt.plot(tuned_params, mean_train_recall_deceased, label = "Train Recall for Deceased", marker = 'o')
 
    plt.xlabel(params)
    plt.ylabel("Recall for 'Deceased'")
    model_title = model + " Model Recall for 'Deaceased' Outcome vs tuned paramters"
    plt.title(model_title)
 
    plt.xticks(rotation=90)        
    plt.tight_layout()     
    plt.legend()
    recall_deceased_save = '../plots/' + model + '_Recall_deceased_plot.png'
    plt.savefig(recall_deceased_save)
 
    plt.figure() #recall
    plt.plot(tuned_params, mean_test_recall, label = "Test Recall", marker = 'o')
    plt.plot(tuned_params, mean_train_recall, label = "Train Recall", marker = 'o')
 
    plt.xlabel(params)
    plt.ylabel("Overall Recall")
    model_title = model + " Model Model Overall Recall vs tuned paramters"
    plt.title(model_title)
 
    plt.xticks(rotation=90)        
    plt.tight_layout()     
    plt.legend()
 
    recall_save = '../plots/' + model + '_Recall_plot.png'
    plt.savefig(recall_save)
 
    plt.show()

################ LGBM MODEL ################

def overfitting_check(x_train, y_train, x_val, y_val, test_data):
    print("Checking for Overfitting...")
    #From https://machinelearningmastery.com/overfitting-machine-learning-models/
    train_scores, test_scores = list(), list()
    n_est = [500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500]
    for d in n_est:
        lg = lgbm.LGBMClassifier(n_estimators= d)
        model = lg.fit(x_train, y_train)        
        train_y_pred = model.predict(x_train)
        train_acc = accuracy_score(y_train, train_y_pred)
        train_scores.append(train_acc)
        test_y_pred = model.predict(x_val)
        test_acc = accuracy_score(y_val, test_y_pred)
        test_scores.append(test_acc)
        if d == None:
            print('>n_estimator: %s, train: %.3f, test: %.3f' % ('None', train_acc, test_acc))
        else:
            print('>n_estimator: %d, train: %.3f, test: %.3f' % (d, train_acc, test_acc))
    
    plt.plot(n_est, train_scores, '-o', label='Train')
    plt.plot(n_est, test_scores, '-o', label='Test')
    plt.title("LGBM Classifier Overfitting")
    plt.xlabel("n_estimator value for LGBM Classifier")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

def LGBMModelSave(x_train, y_train, x_val, y_val, test_data):
    # from https://www.datacamp.com/community/tutorials/adaboost-classifier-python
    print("Running LGBM...")
    lg = lgbm.LGBMClassifier()
    model = lg.fit(x_train, y_train)

    print("Saving LGBM Model...")
    # From https://stackabuse.com/scikit-learn-save-and-restore-models/
    file_pth = os.path.relpath(BASE_DATA + pkl_lgbm_filename, BASE_PATH)
    with open(file_pth, 'wb') as file:
        pickle.dump(model, file)          

def runLGBM(x_train, y_train, x_val, y_val, test_data):
    # x_train, y_train, x_val, y_val, _ = splitForModel(test_data, train_data, val_data)
    print("Running LGBM...")
    model = loadModel(pkl_lgbm_filename)
    y_pred = model.predict(x_val)

    print("y_pred unique:", np.unique(y_pred))
    unique_labels = np.unique(y_val)

    print(unique_labels)
    cf = multilabel_confusion_matrix(y_val, y_pred, labels = unique_labels)
    outputConfusionMatrixMetrics(cf, unique_labels)

def runLGBM_hypertuned(x_train, y_train, x_val, y_val, test_data):
    print("Running LGBM Hypertuned...")
    clf = loadModel(pkl_lgbm_filename)

    custom_scoring = {"recall":make_scorer(recall_score, average = 'macro'),
           "recall_Deceased":  make_scorer(recall_score, labels = [0],  average= None),
           "Accuracy": make_scorer(accuracy_score),
           "f1_score_decease": make_scorer(f1_score, labels = [0], average = None)}

    gridParams_best = {
        'learning_rate': [0.1],
        'num_leaves': [81, 91],
        'max_bin': [100, 120],
        'boosting_type': ['gbdt'],
        'n_estimators': [100, 200, 300],
    }

    gridParams2 = {
        'objective': ['multiclass'],
        'num_class' : [4],
        'learning_rate': [0.1, 0.025],
        'num_leaves': [81, 91],
        'max_bin': [100, 120],
        'boosting_type': ['gbdt'],
        'n_estimators': [100, 200, 300],
    }

    lgbm_grid = GridSearchCV(clf, gridParams2, verbose = 3, cv = 5, refit = False, scoring = custom_scoring, return_train_score= True) #changed to 3 fold to make it faster
    model = lgbm_grid.fit(x_train, y_train)
    pd.DataFrame(lgbm_grid.cv_results_).to_csv("../results/LGBM_tuning2.csv")
    display(pd.DataFrame(lgbm_grid.cv_results_))

def lgbm_predict(x_train, y_train, x_val, y_val, test_data):
 
    best_paramaters = {'boosting_type': 'gbdt', 'learning_rate': 0.1, 'max_bin': 120, 'n_estimators': 300, 'num_leaves': 81}
    test_cpy = test_data.copy()
    print(test_cpy.shape)
    print(x_train.shape)
 
    lg = lgbm.LGBMClassifier(boosting_type= 'gbdt', learning_rate = 0.1, max_bin = 120, n_estimators = 300, num_leaves =  81)
    model = lg.fit(x_train, y_train)
    print("Beginning to predict test data...")
    res_data = model.predict(test_cpy)
    result_df = pd.DataFrame(res_data)
    print(result_df.shape)
    result_df.replace({0:'deceased', 1:'hospitalized', 2:'nonhospitalized', 3:'recovered'},inplace=True)
    np.savetxt('../results/predictions.txt', result_df.values, fmt='%s')
    filename = "../results/predictions.txt"
 
    with open(filename) as f_input:
        data = f_input.read().rstrip('\n')
 
    with open(filename, 'w') as f_output:    
        f_output.write(data)

################ LINEAR SVC MODEL ################
 
def linearSVMModelSave(x_train, y_train, x_val, y_val):
    # from https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html
    # from https://stackoverflow.com/questions/31617530/multiclass-linear-svm-in-python-that-return-probability
    
    print("Running linear SVC...")
    clf = OneVsRestClassifier(LinearSVC(dual=False))
    model = clf.fit(x_train, y_train)
    
    print("Saving linear SVC Model...")
    # From https://stackabuse.com/scikit-learn-save-and-restore-models/
    file_pth = os.path.relpath(BASE_DATA + pkl_linearSVC_filename, BASE_PATH)
    with open(file_pth, 'wb') as file:
        pickle.dump(model, file)  
 
def runLinearSVCClassifier(x_train, y_train, x_val, y_val):
    model = loadModel(pkl_linearSVC_filename)
    y_pred = model.predict(x_val)
    unique_labels = np.unique(y_val)
    cf = multilabel_confusion_matrix(y_val, y_pred, labels = unique_labels)
    outputConfusionMatrixMetrics(cf, unique_labels)
 
def overfittingLinearSVCcheck(x_train, y_train, x_val, y_val):
    print("Checking for Linear SVC Overfitting...")
    #From https://machinelearningmastery.com/overfitting-machine-learning-models/
    train_scores, test_scores = list(), list()
    C_value = [0.1, 0.5, 1, 2, 5, 10, 20, 30, 50, 100]
    for c in C_value:
        clf = OneVsRestClassifier(LinearSVC(C=c, dual=False))
        model = clf.fit(x_train, y_train)
        train_y_pred = model.predict(x_train)
        train_acc = accuracy_score(y_train, train_y_pred)
        train_scores.append(train_acc)
        test_y_pred = model.predict(x_val)
        test_acc = accuracy_score(y_val, test_y_pred)
        test_scores.append(test_acc)
        print('C value: %f, train: %.3f, test: %.3f' % (c, train_acc, test_acc))
    
    plt.plot(C_value, train_scores, '-o', label='Train')
    plt.plot(C_value, test_scores, '-o', label='Test')
    plt.title("Linear SVC Overfitting")
    plt.xlabel("C value")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
 
def runLinearSVC_hypertuned(x_train, y_train, x_val, y_val):
    clf = loadModel(pkl_linearSVC_filename)
    gridParams = {
        "estimator__C" : [0.01, 0.1, 1],
        "estimator__tol" : [0.00001, 0.0001, 0.001],
        "estimator__class_weight" : ['balanced', None]
    }
 
    custom_scoring = {
        "recall":make_scorer(recall_score, average = 'macro'),
        "recall_Deceased":  make_scorer(recall_score, labels = [0],  average= None),
        "Accuracy": make_scorer(accuracy_score),
        "f1_score_decease": make_scorer(f1_score, labels = [0], average = None)
    }
 
    grid = GridSearchCV(clf, gridParams, verbose = 3, cv = 5, refit = False, scoring = custom_scoring, return_train_score= True)
    model = grid.fit(x_train, y_train)
    pd.DataFrame(grid.cv_results_).to_csv("../results/linearSVC_tuning.csv")
    display(pd.DataFrame(grid.cv_results_))

################ RANDOM FOREST MODEL ################

def loadForest():
    file_pth = os.path.relpath(BASE_DATA + pkl_ranforest_filename, BASE_PATH)
    pickle_model = None
    with open(file_pth, 'rb') as file:
        pickle_model = pickle.load(file)
        return pickle_model
    return None
 
def randomForestModel(x_train, y_train, x_val, y_val, test_data):
    print("entering forest")
    file_pth = os.path.relpath(BASE_DATA + pkl_ranforest_filename, BASE_PATH)
    model = RandomForestClassifier(n_estimators = 1)        
    model = model.fit(x_train, y_train)
    with open(file_pth, 'wb') as file:
        pickle.dump(model, file)   
 
def runRandomForestClassifier(x_train, y_train, x_val, y_val, test_data):
    model = loadForest()
    if (model==None):
        print("ERROR: model loaded Nothing")
    else:
        y_pred = model.predict(x_val)
        cf = confusion_matrix(y_val, y_pred)
        unique_labels = np.unique(y_val)
        outputConfusionMatrixMetrics(cf, unique_labels)

def random_forest_test_hparam(x_train, y_train, x_val, y_val, test_data):

    ####Parameters
    max_depth = [x for x in range(20, 35, 5) ]
    n_estimators = [x for x in range(50, 175, 50) ]
    min_samples_split = [2, 4]
    # max_depth = [55]
    # n_estimators = [75, 100]
    # min_samples_split = [2]


    rf = RandomForestClassifier()

    ###################################### now finding the best value ############################
    #GRIDSCORE FINDING THE OPTIMAL VALUES
    param_grid = { 'n_estimators': n_estimators ,
                  'max_depth': max_depth,
                  'min_samples_split': min_samples_split
    }
    # param_grid = { 'n_estimators': [10],
    #               'max_depth': [10],
    #               'min_samples_split': [2]
    # }


    ###################custom scoring ####################
    custom_scoring = {"recall": make_scorer(recall_score, average = "macro"), 
                      "recall_Deceased": make_scorer(recall_score, labels = [0], average=None),
                      "Accuracy": make_scorer(accuracy_score),
                      "f1_score_decease": make_scorer(f1_score, labels = [0], average = None)
                      }
    print("starting grid search...")             
    grid = GridSearchCV(estimator=rf, scoring = custom_scoring, param_grid=param_grid, cv=5, verbose = 3,refit = False, return_train_score=True)
    grid.fit(x_train, y_train) 
    # print("Best parameters for random forests:\n", grid.best_params_) 
    pd.DataFrame(grid.cv_results_).to_csv("../results/Random_Forest_GSCVDepth.csv", index = False)
    print(pd.DataFrame(grid.cv_results_))

#done with best parameter
# {'max_depth': 55, 'min_samples_split': 2, 'n_estimators': 50}

def random_forest_predict(x_train, y_train, test_data): # Testing thinking random forest is best
    outcome_dict = {
        0:'deceased',
        1:'hospitalized',
        2:'nonhospitalized',
        3:'recovered'
    }

    rf = RandomForestClassifier(n_estimators=50, max_depth=55, min_samples_split=2)
    print("beginning to build model...")
    model = rf.fit(x_train, y_train)
    print("Beginning to predict test data...")
    res_data = model.predict(test_data)

    predict_fp = '../results/predictions.txt'
    res_data = np.array([outcome_dict[xi] for xi in res_data])
    np.savetxt(predict_fp, res_data, fmt='%s')
    with open(predict_fp) as f_input:
        data = f_input.read().rstrip('\n')

    with open(predict_fp, 'w') as f_output:
        f_output.write(data)
    check_if_file_valid(predict_fp)
    return        

def plot_ranforest_data():
    df = pd.read_csv('../results/Random_Forest_GSCVDepth.csv')
    f1_deceased_test = df['mean_test_f1_score_decease'].to_numpy()
    f1_deceased_train = df['mean_train_f1_score_decease'].to_numpy()

    recall_deceased_test = df['mean_test_recall_Deceased'].to_numpy()
    recall_deceased_train = df['mean_test_recall_Deceased'].to_numpy()

    accuracy_train = df['mean_test_Accuracy'].to_numpy()
    accuracy_test = df['mean_train_Accuracy'].to_numpy()
    print(accuracy_test)
    
    params = df['params'].to_numpy()
    print("plotting...")
    plt.figure(1)
    plt.title("Random Forest: F1_Score on Deceased ")
    plt.plot(params, f1_deceased_test, color = "blue", label='test')
    plt.plot(params, f1_deceased_train, color = "orange", label = "train")
    plt.xticks(rotation=90)
    plt.legend()
    plt.show()
    
    plt.figure(2)
    plt.title("Random Forest: Recall deceased")
    plt.plot(params, recall_deceased_test, color = "blue")
    plt.plot(params, recall_deceased_train, color = "orange")
    plt.legend()

    plt.figure(3)
    plt.title("Random Forest: Accuracy ")
    plt.plot(params, accuracy_test, color = "blue")
    plt.plot(params, accuracy_train, color = "orange")
    plt.legend()
    return

def check_if_file_valid(filename):
    assert filename.endswith('predictions.txt'), 'Incorrect filename'
    f = open(filename).read()
    l = f.split('\n')
    print("length = ", len(l))
    assert len(l) == 46500, 'Incorrect number of items'
    assert (len(set(l)) == 4), 'Wrong class labels'
    return 'The predictions file is valid'