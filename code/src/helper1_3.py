from main import *

# 1.3 Find outliers 
def find_outliers(train_data):
    print("=========Printing Outlier=========")

    print("\n\nNumerical Data")
    numerical_outliers(train_data)

    print("\n\nCategorical")
    categorical_outliers(train_data)

def numerical_outliers(dataset: pd.DataFrame):
    data_copy = dataset.copy()
    #print( len(test_data['age'] == "43")  )
    # missing_train = train_copy.isnull().sum(axis = 0)
    numeric_cols = data_copy.select_dtypes(include=[np.number, float, int]).columns
    data_zscore = data_copy[numeric_cols].apply(zscore)

    # print(type((np.abs(zscore(train_copy['age']) ) < 3))) 
    #print(train_copy['age'].value_counts() )
    for col in data_zscore:
        Q1 = data_zscore[col].quantile(0.25)
        Q3 = data_zscore[col].quantile(0.75)
        IQR = Q3 - Q1
        filter = lambda score : score >= (Q1 - (1.5*IQR)) and score <= (Q3 + (1.5*IQR))
        score = data_zscore[col].apply(filter)
        print(score.value_counts())
        print()

def categorical_outliers(dataset: pd.DataFrame):
    data_copy = dataset.copy()
    categorical_cols = data_copy.select_dtypes(exclude=[np.number]).columns

    for cols in categorical_cols:
        freq = pd.Series(data_copy[cols].value_counts()).array
        data_zscore = pd.Series(zscore(freq))
        Q1 = data_zscore.quantile(0.25)
        Q3 = data_zscore.quantile(0.75)
        IQR = Q3 - Q1
        filter = lambda score : score >= (Q1 - (1.5*IQR)) and score <= (Q3 + (1.5*IQR))
        score = data_zscore.apply(filter)
        print(score.value_counts())
        print(cols)
        print()
