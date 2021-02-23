from main import *

def data_cleaning_and_missing_values(location_data, test_data, train_data):
    # Case and train data
    train_and_test_methods(test_data)
    train_and_test_methods(train_data)

def train_and_test_methods(dataset):
    replace_missing_with_default_cases(dataset)
    impute_dates(dataset)
    remove_missing_lat_and_long_cases(dataset)
    impute_age(dataset)
    impute_missing_countries_cases(dataset)
    impute_missing_provinces_case(dataset)
    convertDataToFloatsForCases(dataset)
    train_Peurto_Rico_Fix(dataset)

#### CASE TRAIN AND CASE TEST METHODS ####

# Only for Case_train and test_train
def replace_missing_with_default_cases(dataset):
    dataset['sex'] = dataset['sex'].replace(np.nan, 'unknown')
    dataset['additional_information'] = dataset['additional_information'].replace(np.nan, 'unknown')
    dataset['source'] = dataset['source'].replace(np.nan, 'unknown')
    dataset['date_confirmation'] = dataset['date_confirmation'].replace(np.nan, '00.00.0000')

def impute_dates(dataset: pd.DataFrame):
    dataset['date_confirmation'] = dataset['date_confirmation'].astype(str)
    dataset['date_confirmation'] = dataset['date_confirmation'].apply(dateFormatFix)

# Applys uniform dates format to train data 
def dateFormatFix(date):
    new_date = date
    try:
        datetime.strptime(new_date, '%d.%m.%Y')
    except ValueError:
        #either nan or date range so
        date_list = new_date.split(" - ")
        if len(date_list) > 1:
            #perform the operation
            a = datetime.strptime(date_list[0], '%d.%m.%Y')
            b = datetime.strptime(date_list[1], '%d.%m.%Y')
            med_date = a + (b-a) / 2
            return med_date.strftime('%d.%m.%Y')
        elif (len(date_list) > 0):  #
            date_list = new_date.split("-")
            if len(date_list) > 1:
            #perform the operation
                a = datetime.strptime(date_list[0], '%d.%m.%Y')
                b = datetime.strptime(date_list[1], '%d.%m.%Y')
                med_date = a + (b-a) / 2
                return med_date.strftime('%d.%m.%Y')
    return new_date

def remove_missing_lat_and_long_cases(dataset):
    # Imputing latitude and longitude for cases_train
    index = dataset[(dataset['latitude'].isnull()) & (dataset['longitude'].isnull())].index
    dataset.drop(index, inplace=True)
    dataset.reset_index(drop=True, inplace=True)

def impute_age(dataset: pd.DataFrame):
    dataset['age'] = dataset['age'].astype(str)
    dataset['age'] = dataset['age'].apply(ageFormatFix)
    h = dataset.groupby('country', as_index=False)['age'].mean()
    h_dict = h.set_index('country').to_dict()
    h_dict = h_dict['age']
    dataset['age'] = dataset['age'].fillna(dataset['country'].map(h_dict))
    age_mean = int(dataset['age'].mean())
    dataset['age'] = dataset['age'].fillna(age_mean)

def ageFormatFix(age):
    num = []
    s = ""
    for c in age:
        if c.isnumeric() or c==".":
            s+=c
        elif (len(s) > 0):
            num.append(int(float(s)) )
            s = ""
    if (len(s) > 0 ):
        num.append(int(float(s)) )
    if (len(num)==0) :
        return None
    return int(statistics.mean(num))

def country_age_avg(row: pd.Series, d: dict) :
    if row['country'] in d:
        return d[row['country'] ]
    else :
        return None

def impute_missing_countries_cases(dataset):
    # Impute missing countries for cases_train
    dataset['country'] = dataset.apply(lambda x: "China" if pd.isnull(x['country']) else x['country'], axis=1)

def impute_missing_provinces_case(dataset):
    temp = dataset.copy()

    temp['latitude'] = temp['latitude'].astype(int)
    temp['longitude'] = temp['longitude'].astype(int)
    temp['latitude'] = temp['latitude'].astype(str)
    temp['longitude'] = temp['longitude'].astype(str)

    temp['lat-long'] = list(zip(temp.latitude, temp.longitude))

    filtered_data = temp.dropna(subset=['province']).drop_duplicates('lat-long').set_index('lat-long')['province']
    temp['province'] = temp['province'].fillna(temp['lat-long'].map(filtered_data))

    dataset['province'] = temp['province']

    dataset['province'] = dataset['province'].replace(np.nan, 'unknown')

#converting train data values 
def convertDataToFloatsForCases(dataset):
    dataset['latitude'] = dataset['latitude'].astype(float)
    dataset['longitude'] = dataset['longitude'].astype(float)
    dataset['age'] = dataset['age'].astype(int)


def train_Peurto_Rico_Fix(dataset):
    dataset['province'] = dataset.apply(lambda x: "Puerto Rico" if x['country'] == "Puerto Rico" else x['province'], axis=1 )
    dataset['country'] =  dataset.apply(lambda x: "United States" if x['country'] =="Puerto Rico" else x['country'], axis=1 )
