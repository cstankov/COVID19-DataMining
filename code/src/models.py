from main import *

pkl_ada_filename = "ada_pickle_model.pkl"

def loadAda():
    return


def adaBoostModel(train_data, val_data):
    # data_cpy = train_data.copy()
    # from https://www.datacamp.com/community/tutorials/adaboost-classifier-python
    le = preprocessing.LabelEncoder()
    train_cpy = train_data.copy()
    val_cpy = val_data.copy()

    train_cpy = train_cpy.apply(le.fit_transform)
    val_cpy = val_cpy.apply(le.fit_transform)
    # print(train_cpy.columns)
    x_train = train_cpy.loc[:, train_cpy.columns != 'Outcome']
    y_train = train_cpy['Outcome']
    x_val = val_cpy.loc[:, val_cpy.columns != 'Outcome']
    y_val = val_cpy['Outcome']

    x_train = x_train.to_numpy()
    y_train = y_train.to_numpy()
    x_val = x_val.to_numpy()
    y_val = y_val.to_numpy()
   
    # ada = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(max_depth=3),
    #                            algorithm="SAMME.R",
    #                            n_estimators=50, 
    #                            learning_rate=1, 
    #                            random_state = 0) # 55 to 59 percent
   
    # 57 to 62 percent
    # ada = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(max_depth=2),
    #                            algorithm="SAMME.R", n_estimators = 200, learning_rate = 0.2)
    # model = ada.fit(x_train, y_train)
    sum_acc = 0
    for i in range(0, 5): 
        # 40 to 50 percent
        # lgbm = LGBMClassifier(n_estimators = 100, 
        #                     learning_rate = 0.7)
        # model = lgbm.fit(x_train, y_train) 

        # percent
        ada = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(max_depth=2),
                               algorithm="SAMME.R", n_estimators = 200, learning_rate = 0.2)
        model = ada.fit(x_train, y_train)

        #56 percent
        # ada = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(max_depth=2),
        #                        algorithm="SAMME.R", n_estimators = 50, learning_rate = 1, random_state = 0)
        # model = ada.fit(x_train, y_train)

        # 45.3 percent
        # cbc = CatBoostClassifier(depth=10, iterations= 500, l2_leaf_reg= 9, learning_rate= 0.15)
        # model = cbc.fit(x_train, y_train)


        y_pred = model.predict(x_val)
        sum_acc += metrics.accuracy_score(y_val, y_pred)
    
    avg_acc = sum_acc / 5

    print("Average Accuracy: ", avg_acc)                    

