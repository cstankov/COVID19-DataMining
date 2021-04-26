from main import *
from dataHandler import *

pkl_lgbm_filename = "lgbm_pickle_model.pkl"
pkl_linearSVC_filename = "linearSVC_pickle_model.pkl"
pkl_ranforest_filename = "ranforest_pickle_model.pkl"
BASE_PATH = os.path.dirname(__file__)
BASE_DATA = '../models/'   
    
def splitForModel(test_data, train_data, val_data, return_numpy = True, convert_categorical = True):
    print("Splitting Model...")
    # main_train_data = pd.read_csv("../results/cases_train_processed.csv")
    train_cpy = train_data.copy()
    val_cpy = val_data.copy()
    test_cpy = test_data.copy()

    #print(np.unique(train_cpy['Case-Fatality_Ratio']))
    train_cpy.drop(['Source'], axis = 1, inplace = True)
    val_cpy.drop(['Source'], axis = 1, inplace = True)
    test_cpy.drop(['Source'], axis = 1, inplace = True)
    
    print("Doing for OUTCOME")
    train_cpy['Outcome'].replace({'deceased':0, 'hospitalized':1, 'nonhospitalized':2, 'recovered':3},inplace=True)
    val_cpy['Outcome'].replace({'deceased':0, 'hospitalized':1, 'nonhospitalized':2, 'recovered':3},inplace=True)

    categorical_cols = ["Sex",  "Province_State", "Country", "Combined_Key", "Date_Confirmation"]


    enc_dict = {}
    count = 0
    #print(np.unique(train_cpy['Case-Fatality_Ratio']))
    if convert_categorical == True:
        for col in categorical_cols:

            train_cpy[col] = train_cpy[col].apply(lambda x: int(hashlib.sha1(x.encode("utf-8")).hexdigest(), 16)) % (10 ** 8)
            val_cpy[col] = val_cpy[col].apply(lambda x: int(hashlib.sha1(x.encode("utf-8")).hexdigest(), 16)) % (10 ** 8)
            test_cpy[col] = test_cpy[col].apply(lambda x: int(hashlib.sha1(x.encode("utf-8")).hexdigest(), 16)) % (10 ** 8)

    
    print("Done categorical encoding...")  
    # print(train_cpy.head(30))  
    x_train = train_cpy.loc[:, train_cpy.columns != 'Outcome']
    # print("unique sex values")
    # print(np.unique(train_cpy['Sex']))

    # print("x_train")
    # print(x_train.head(10))
    y_train = train_cpy['Outcome']
    x_val = val_cpy.loc[:, val_cpy.columns != 'Outcome']
    y_val = val_cpy['Outcome']


    #     for col in df_num:
    #     print(df_num.groupby(np.isinf(df_num[col])).count())
    # train_data, val_data = split_train_val(train_data_processed)


    if return_numpy == True:
        x_train = x_train.to_numpy()
        y_train = y_train.to_numpy()
        x_val = x_val.to_numpy()
        y_val = y_val.to_numpy()

    return x_train, y_train, x_val, y_val, test_data


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

    if ModelFileName == pkl_lgbm_filename:
        print("Loading LGBM Model...")
    elif ModelFileName == pkl_linearSVC_filename:
        print("Loading SVC Model...")

    with open(file_pth, 'rb') as file:
        pickle_model = pickle.load(file)
    return pickle_model 

################ LGBM MODEL ################
def overfitting_check(x_train, y_train, x_val, y_val, test_data):
    print("Checking for Overfitting...")
    #From https://machinelearningmastery.com/overfitting-machine-learning-models/
    # x_train, y_train, x_val, y_val, _ = splitForModel(test_data, train_data, val_data)
    train_scores, test_scores = list(), list()

    n_est = [500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500]
    for d in n_est:
        lg = lgbm.LGBMClassifier(n_estimators= d)
        model = lg.fit(x_train, y_train)        
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
 
    # x_train, y_train, x_val, y_val, _ = splitForModel(test_data, train_data, val_data)
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
    # x_train, y_train, x_val, y_val, _ = splitForModel(test_data, train_data, val_data)
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
    'learning_rate': [0.1],
    'num_leaves': [71, 81],
    'max_bin': [91, 101],
    'boosting_type': ['gbdt'],
    'n_estimators': [300, 450],
    }

    lgbm_grid = GridSearchCV(clf, gridParams, verbose = 3, cv = 5, refit = False, scoring = custom_scoring, return_train_score= True)
    model = lgbm_grid.fit(x_train, y_train)
    pd.DataFrame(lgbm_grid.cv_results_).to_csv("../results/LGBM_tuning.csv")
    display(pd.DataFrame(lgbm_grid.cv_results_))
    
    # grid_search_best_params = {'boosting_type': 'gbdt', 'learning_rate': 0.1, 'max_bin': 100, 'n_estimators': 300, 'num_leaves': 81}

    # lg = lgbm.LGBMClassifier(boosting_type='gbdt', learning_rate= 0.1, max_bin= 100, n_estimators= 300, num_leaves= 81)
    # model = lg.fit(x_train, y_train)
    # y_pred = model.predict(x_val)
    # unique_labels = np.unique(y_val)
    # print(unique_labels)
    # cf = multilabel_confusion_matrix(y_val, y_pred, labels = unique_labels)
    # outputConfusionMatrixMetrics(cf, unique_labels)

def plot_result(model_name, model_result_path, x_train, y_train, x_val, y_val, test_data):
    if (model_name == 'LGBM'):
        if not (os.path.exists(model_result_path)):
            runLGBM_hypertuned(x_train, y_train, x_val, y_val, test_data)


        lgbm_result = pd.read_csv(model_result_path)

        tuned = list(zip(lgbm_result.param_num_leaves, lgbm_result.param_n_estimators, lgbm_result.param_max_bin))
        tuned_params = ['{}, {}, {}'.format(x,y,z) for x,y,z in tuned]
        print("tuned_params: ")
        print(tuned_params)
        # lgbm_num_leaves = lgbm_result['param_num_leaves']
        # lgbm_n_estimators = lgbm_result['param_n_estimators']
        # lgbm_max_bin = lgbm_result['param_max_bin']

        lgbm_mean_test_accuracy = list(lgbm_result['mean_test_Accuracy'])
        lgbm_mean_train_accuracy = list(lgbm_result['mean_train_Accuracy'])

        lgbm_mean_test_f1_deceased = list(lgbm_result['mean_test_f1_score_decease'])
        lgbm_mean_train_f1_deceased = list(lgbm_result['mean_train_f1_score_decease'])

        lgbm_mean_test_recal = list(lgbm_result['mean_test_recall'])
        lgbm_mean_train_recal = list(lgbm_result['mean_train_recall'])

        lgbm_mean_test_recall_deceased = list(lgbm_result['mean_test_recall_Deceased'])
        lgbm_mean_train_recall_deceased = list(lgbm_result['mean_train_recall_Deceased'])
        # print(lgbm_mean_test_accuracy)

        plt.figure() #accuracy
        plt.plot(tuned_params, lgbm_mean_test_accuracy, label = "Test Accuracy", marker = 'o')
        plt.plot(tuned_params, lgbm_mean_train_accuracy, label = "Train Accuracy", marker = 'o')

        plt.xlabel("(num_leaves, n_etimators, max_bin)")
        plt.ylabel("Accuracy")
        plt.title("LGBM Model Accuracy vs tuned paramters")

        plt.xticks(rotation=90)   
        plt.tight_layout()     
        plt.legend()

        plt.savefig('../plots/LGBM_accuracy_plot.png')

        plt.figure() #f1-score-deceased
        plt.plot(tuned_params, lgbm_mean_test_f1_deceased, label = "Test F1 Score for Deceased", marker = 'o')
        plt.plot(tuned_params, lgbm_mean_train_f1_deceased, label = "Train F1 Score for Deceased", marker = 'o')

        plt.xlabel("(num_leaves, n_etimators, max_bin)")
        plt.ylabel("F1 SCore for 'Deceased'")
        plt.title("LGBM Model F1 Score for 'Deaceased' Outcome vs tuned paramters")

        plt.xticks(rotation=90)        
        plt.tight_layout()     
        plt.legend()

        plt.savefig('../plots/LGBM_F1_score_deceased_plot.png')

        plt.figure() #recall-deceased
        plt.plot(tuned_params, lgbm_mean_test_recall_deceased, label = "Test Recall for Deceased", marker = 'o')
        plt.plot(tuned_params, lgbm_mean_train_recall_deceased, label = "Train Recall for Deceased", marker = 'o')

        plt.xlabel("(num_leaves, n_etimators, max_bin)")
        plt.ylabel("Recall for 'Deceased'")
        plt.title("LGBM Model Recall for 'Deaceased' Outcome vs tuned paramters")

        plt.xticks(rotation=90)        
        plt.tight_layout()     
        plt.legend()

        plt.savefig('../plots/LGBM_Recall_deceased_plot.png')

        plt.figure() #recall
        plt.plot(tuned_params, lgbm_mean_test_recal, label = "Test Recall", marker = 'o')
        plt.plot(tuned_params, lgbm_mean_train_recal, label = "Train Recall", marker = 'o')

        plt.xlabel("(num_leaves, n_etimators, max_bin)")
        plt.ylabel("Overall Recall")
        plt.title("LGBM Model Overall Recall vs tuned paramters")

        plt.xticks(rotation=90)        
        plt.tight_layout()     
        plt.legend()

        plt.savefig('../plots/LGBM_Recall_plot.png')

        

        plt.show()

        # plt.savefig()

    elif model_name == 'LinearSVC':
         if not (os.path.exists(model_result_path)):
             runLinearSVC_hypertuned(test_data, train_data = train_data, val_data = val_data)
    else: 
        print("Invalid Model name given. Expected 'LGBM', 'LinearSVC' or 'RandomForest', got: ", model_name)

     # elif model_name == 'RandomForest':
    #     if not (os.path.exists(model_result_path)):
    #         runRandomForest_hypertuned(train_data=train_data, val_data=val_data)
    #  

################ LINEAR SVC MODEL ################
 
def linearSVMModelSave(x_train, y_train, x_val, y_val):
    # from https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html
    # from https://stackoverflow.com/questions/31617530/multiclass-linear-svm-in-python-that-return-probability
    # x_train, y_train, x_val, y_val = splitForModel(train_data, val_data)
    
    print("Running linear SVC...")
    clf = OneVsRestClassifier(LinearSVC(dual=False))
    model = clf.fit(x_train, y_train)
    
    print("Saving linear SVC Model...")
    # From https://stackabuse.com/scikit-learn-save-and-restore-models/
    file_pth = os.path.relpath(BASE_DATA + pkl_linearSVC_filename, BASE_PATH)
    with open(file_pth, 'wb') as file:
        pickle.dump(model, file)  
 
def runLinearSVCClassifier(x_train, y_train, x_val, y_val):
    # x_train, y_train, x_val, y_val = splitForModel(train_data, val_data)
    model = loadModel(pkl_linearSVC_filename)
 
    y_pred = model.predict(x_val)
 
    unique_labels = np.unique(y_val)
 
    cf = multilabel_confusion_matrix(y_val, y_pred, labels = unique_labels)
    outputConfusionMatrixMetrics(cf, unique_labels)
 
def overfittingLinearSVCcheck(x_train, y_train, x_val, y_val):
    print("Checking for Linear SVC Overfitting...")
    #From https://machinelearningmastery.com/overfitting-machine-learning-models/
    # x_train, y_train, x_val, y_val = splitForModel(train_data, val_data)
    train_scores, test_scores = list(), list()
    C_value = [0.1, 0.5, 1, 2, 5, 10, 20, 30, 50, 100]
    for c in C_value:
        clf = OneVsRestClassifier(LinearSVC(C=c, dual=False))
        model = clf.fit(x_train, y_train)
         # evaluate on the train dataset
        train_y_pred = model.predict(x_train)
        train_acc = accuracy_score(y_train, train_y_pred)
        train_scores.append(train_acc)
        # evaluate on the test dataset
        test_y_pred = model.predict(x_val)
        test_acc = accuracy_score(y_val, test_y_pred)
        test_scores.append(test_acc)
        # summarize progress
        print('C value: %f, train: %.3f, test: %.3f' % (c, train_acc, test_acc))
    
    plt.plot(C_value, train_scores, '-o', label='Train')
    plt.plot(C_value, test_scores, '-o', label='Test')
    plt.title("Linear SVC Overfitting")
    plt.xlabel("C value")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
 
def runLinearSVC_hypertuned(x_train, y_train, x_val, y_val):
    # x_train, y_train, x_val, y_val = splitForModel(train_data, val_data)
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
    # x_train, y_train, x_val, y_val, _= splitForModel(test_data, train_data, val_data)
    model = RandomForestClassifier(n_estimators = 1)        #Please remove this parameter here to get default model of 100 trees
    model = model.fit(x_train, y_train)
    with open(file_pth, 'wb') as file:
        pickle.dump(model, file)  
 
 
    #GRIDSCORE FINDING THE OPTIMAL VALUES
    # param_grid = { 'n_estimators': [200, 400, 500],
    #               'max_features': ['auto', 'log2'],
    #               'oob_score': [True]
    # }
    # grid = GridSearchCV(model, param_grid, refit = True, verbose = 3,n_jobs=-1)
    # grid.fit(x_train, y_train) 
    # # print best parameter after tuning 
    # print("oob score= ", grid.oob_score)
    # print("score = ", grid.score)
    # print(grid.best_params_) 
    # grid_pred = grid.predict(x_val) 
 
 
 
def runRandomForestClassifier(x_train, y_train, x_val, y_val, test_data):
    # x_train, y_train, x_val, y_val, _= splitForModel(test_data, train_data, val_data)
    model = loadForest()
    if (model==None):
        print("ERROR: model loaded Nothing")
    else:
        y_pred = model.predict(x_val)
        cf = confusion_matrix(y_val, y_pred)
        unique_labels = np.unique(y_val)
        outputConfusionMatrixMetrics(cf, unique_labels)
        # x_test, y_test, bleh, bleh2 = splitForModel(train_data, val_data)
        # score = []
        # test_score = []
        # print("starting to train")
        # for i in range(1,11,1):
        #     clf = RandomForestClassifier(n_estimators = i)
        #     print('running estimator = ', i)
        #     model = clf.fit(x_train, y_train)
        #     print("starting to score...")
        #     score1= model.score(x_train, y_train)
        #     score2=model.score(x_val, y_val)
        #     print(f"score1 = {score1} score2 = {score2}")
        #     score.append(score1)
        #     test_score.append(score2)
        #     #file_pth = os.path.relpath(BASE_DATA + pkl_ranforest_filename, BASE_PATH)
        #     # y_pred = model.predict(x_val)
        #     # cf = confusion_matrix(y_val, y_pred)
        #     # outputConfusionMatrixMetrics(cf)
 
        # num_trees = [_ for _ in range(1,11,1) ]
        # plt.plot(num_trees, score, 'r', num_trees, test_score, 'b')
        # plt.xlabel('number of trees')
        # plt.ylabel('Accuracy')
        # plt.show()

def random_forest_test_hparam(x_train, y_train, x_val, y_val, test_data):

    ####Parameters
    max_depth = [x for x in range(10, 30, 5) ]
    n_estimators = [x for x in range(50, 175, 50) ]
    min_samples_split = [2, 4]


    rf = RandomForestClassifier(max_depth=15, n_estimators=50, min_samples_split=2)
    rf.fit(x_train, y_train)
    print("FIT DONE")
    return

    ###################################### now finding the best value ############################
    #GRIDSCORE FINDING THE OPTIMAL VALUES
    param_grid = { 'n_estimators': n_estimators ,
                  'max_depth': max_depth,
                  'min_samples_split': min_samples_split
    }

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
    pd.DataFrame(grid.cv_results_).to_csv("Random_Forest_GSCVDepth.csv")
    print(pd.DataFrame(grid.cv_results_))

#done with best parameter
# {'max_depth': 25, 'min_samples_split': 2, 'n_estimators': 50}

def random_forest_test(x_train, y_train, x_val, y_val, test_data): # Testing thinking random forest is best
    # x_train, y_train, x_val, y_val ,le = splitForModel(train_data, val_data )
    return
    le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print("Mapping:")
    print(le_name_mapping)
    print("\n\n")
    test_cpy = test_data.loc[:, test_data.columns != 'Outcome']
    test_cpy = test_cpy.apply(le.fit_transform)
    # print(test_cpy.head(20) )
    test_cpy = test_cpy.to_numpy()

    le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print("Mapping:")
    print(le_name_mapping)
    print("\n\n")

    rf = RandomForestClassifier(n_estimators=50, max_depth=25, min_samples_split=2)
    print("beginning to build model...")
    print(le.inverse_transform(y_train) [5])
    model = rf.fit(x_train, y_train)
    print("Beginning to predict test data...")
    res_data = model.predict(x_train)
    pd.DataFrame(res_data).to_csv("test_data_predictionsNum.csv")
    res_data = le.inverse_transform(res_data)
    print("test data prediction complete")
    pd.DataFrame(res_data).to_csv("test_data_predictions.csv")
    return        


def lgbm_predict(x_train, y_train, x_val, y_val, test_data):
    best_paramaters = {'boosting_type': 'gbdt', 'learning_rate': 0.1, 'max_bin': 120, 'n_estimators': 300, 'num_leaves': 81}
    test_cpy = test_data.loc[:, test_data.columns != 'Outcome']

    lg = lgbm.LGBMClassifier(boosting_type= 'gbdt', learning_rate = 0.1, max_bin = 120, n_estimators = 300, num_leaves =  81)
    model = lg.fit(x_train, y_train)
    print("Beginning to predict test data...")
    res_data = model.predict(x_train)
    result_df = pd.DataFrame(res_data)
    result_df.replace({0:'deceased', 1:'hospitalized', 2:'nonhospitalized', 3:'recovered'},inplace=True)
    # result_df = 
    np.savetxt(r'../results/prediction.txt', result_df.values, fmt='%s')

 def check_if_file_valid(filename):
    assert filename.endswith('predictions.txt'), 'Incorrect filename'
    f = open(filename).read()
    l = f.split('\n')
    assert len(l) == 46500, 'Incorrect number of items'
    assert (len(set(l)) == 4), 'Wrong class labels'
    return 'The predictions file is valid'