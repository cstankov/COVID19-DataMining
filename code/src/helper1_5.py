from main import * 

def joining_datasets(location_data, test_data, train_data):
    test_data_processed = cases_join_locaction(test_data, location_data, True)
    train_data_processed = cases_join_locaction(train_data, location_data)
    # test_data_processed = train_data_processed
    return test_data_processed, train_data_processed


def addCombinedKey(dataset):
    dataset['province'] = dataset['province'].str.strip()
    dataset['country'] = dataset['country'].str.strip()
    dataset.insert(9, 'Combined_Key', dataset['province'] + ', ' + dataset['country'])
    return dataset


def fixNan(dataset):
    dataset['province'] = dataset.apply(lambda x: x['province'] if pd.isna(x['Province_State']) else x['Province_State'], axis = 1)
    dataset['country'] = dataset.apply(lambda x: x['country'] if pd.isna(x['Country_Region']) else x['Country_Region'], axis = 1)

    dataset.insert(4, 'Latitude', None)
    dataset.insert(5, 'Longitude', None)
    dataset['Latitude'] = dataset.apply(lambda x: x['Lat'] if pd.isna(x['latitude']) else x['latitude'], axis = 1)
    dataset['Longitude'] = dataset.apply(lambda x: x['Long_'] if pd.isna(x['longitude']) else x['longitude'], axis = 1)
    return dataset


def dropAdditionalColumns(dataset):
    dataset = dataset.drop(['latitude', 'longitude', 'Lat', 'Long_'], axis=1)
    dataset = dataset.drop(['Province_State', 'Country_Region'], axis=1)
    dataset = dataset.drop(['Last_Update', 'additional_information'], axis=1)
    return dataset

def rearrangeColumns(dataset):
    cols = dataset.columns.to_list()
    cols = ['age', 'sex', 'province', 'country', 'Combined_Key', 'Latitude', 'Longitude', 'date_confirmation', 'Confirmed', 'Deaths', 'Recovered', 'Active', 'Incidence_Rate', 'Case-Fatality_Ratio', 'source', 'outcome']
    dataset = dataset[cols]
    return dataset


def standardizeColumns(dataset, is_test):
    if is_test == False:
        dataset = dataset[dataset['outcome'].notna()] # Droping where outcome is Nan for train only
    rename_cols = ['Age', 'Sex', 'Province_State', 'Country', 'Combined_Key', 'Latitude', 'Longitude', 'Date_Confirmation', 'Confirmed', 'Deaths', 'Recovered', 'Active', 'Incidence_Rate', 'Case-Fatality_Ratio', 'Source', 'Outcome']
    dataset.columns = rename_cols
    return dataset

def imputeMissingValues(dataset):
    dataset['Confirmed'] = dataset['Confirmed'].replace(np.nan, -1.0)
    dataset['Deaths'] = dataset['Deaths'].replace(np.nan, -1.0)
    dataset['Recovered'] = dataset['Recovered'].replace(np.nan, -1.0)
    dataset['Active'] = dataset['Active'].replace(np.nan, -1.0)
    dataset['Incidence_Rate'] = dataset['Incidence_Rate'].replace(np.nan, -1.0)
    dataset['Case-Fatality_Ratio'] = dataset['Case-Fatality_Ratio'].replace(np.nan, -1.0)
    
    return dataset


def cases_join_locaction(dataset, location_data, is_test=False):
    temp3 = dataset.copy()
    temp3 = addCombinedKey(temp3)

    temp_train = temp3.copy()
    temp_loc = location_data.copy()

    merged = pd.merge(temp_train,temp_loc, on=['Combined_Key'], how='left')
    
    merged = fixNan(merged)

    # # remove other lat and long columns
    merged = dropAdditionalColumns(merged)

    merged = rearrangeColumns(merged)

    merged_data = merged.copy()
    merged_data = standardizeColumns(merged_data, is_test)

    merged_data = imputeMissingValues(merged_data)
    return merged_data
