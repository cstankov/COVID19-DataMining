from main import *

# 1.3 Find outliers 
def find_outliers(location_data, test_data, train_data):
    numerical_outliers(location_data)
    numerical_outliers(train_data)
    # numerical_outliers(test_data)
    categorical_outliers(location_data)
    categorical_outliers(train_data)
    # categorical_outliers(test_data)
    drop_neg_active(location_data)
    remove_not_follow_active_formula(location_data)

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

# 1.3 Dealing with negative active cases outliers
def drop_neg_active(location_data):
    location_data = location_data.drop(location_data[location_data['Active'] < 0.0].index)
    location_data.reset_index(drop=True, inplace=True)


# 1.3 Dropping rows that dont follow this formula: Active = confrimed - deaths - recovered 
def remove_not_follow_active_formula(location_data: pd.DataFrame):
    location_data = location_data.drop(location_data[location_data['Active'] != (location_data['Confirmed'] - location_data['Deaths'] - location_data['Recovered'])].index)
    location_data.reset_index(drop=True, inplace=True)

