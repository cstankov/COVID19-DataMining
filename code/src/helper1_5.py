from main import * 

def joining_datasets(location_data, test_data, train_data):
    test_data_processed = cases_join_locaction(test_data, location_data, True)
    train_data_processed = cases_join_locaction(train_data, location_data)
    # train_data_processed = test_data_processed
    # print(test_data[:20])
    # print(test_data_processed[:10])
    return test_data_processed, train_data_processed

def cases_join_locaction(dataset, location_data, is_test=False):
    temp3 = dataset.copy()
    temp3['province'] = temp3['province'].str.strip()
    temp3['country'] = temp3['country'].str.strip()
    temp3.insert(9, 'Combined_Key', temp3['province'] + ', ' + temp3['country'])

    # print(temp3['Combined_Key'].equals(location_data['Combined_Key']))

    temp_train = temp3.copy()
    temp_loc = location_data.copy()

    merged = pd.merge(temp_train,temp_loc, on=['Combined_Key'], how='left')
    merged['province'] = merged.apply(lambda x: x['province'] if pd.isna(x['Province_State']) else x['Province_State'], axis = 1)
    merged['country'] = merged.apply(lambda x: x['country'] if pd.isna(x['Country_Region']) else x['Country_Region'], axis = 1)

    merged.insert(4, 'Latitude', None)
    merged.insert(5, 'Longitude', None)
    merged['Latitude'] = merged.apply(lambda x: x['Lat'] if pd.isna(x['latitude']) else x['latitude'], axis = 1)
    merged['Longitude'] = merged.apply(lambda x: x['Long_'] if pd.isna(x['longitude']) else x['longitude'], axis = 1)


    # # remove other lat and long columns
    merged = merged.drop(['latitude', 'longitude', 'Lat', 'Long_'], axis=1)
    merged = merged.drop(['Province_State', 'Country_Region'], axis=1)
    merged = merged.drop(['Last_Update', 'additional_information'], axis=1)


    cols = merged.columns.to_list()
    cols = ['age', 'sex', 'province', 'country', 'Combined_Key', 'Latitude', 'Longitude', 'date_confirmation', 'Confirmed', 'Deaths', 'Recovered', 'Active', 'Incidence_Rate', 'Case-Fatality_Ratio', 'source', 'outcome']
    merged = merged[cols]

    merged_data = merged.copy()
    if is_test == False:
        merged_data = merged_data[merged_data['outcome'].notna()] # Droping 65,279 rows (about 17.7%)
    rename_cols = ['Age', 'Sex', 'Province_State', 'Country', 'Combined_Key', 'Latitude', 'Longitude', 'Date_Confirmation', 'Confirmed', 'Deaths', 'Recovered', 'Active', 'Incidence_Rate', 'Case-Fatality_Ratio', 'Source', 'Outcome']
    merged_data.columns = rename_cols

    merged_data['Confirmed'] = merged_data['Confirmed'].replace(np.nan, -1.0)
    merged_data['Deaths'] = merged_data['Deaths'].replace(np.nan, -1.0)
    merged_data['Recovered'] = merged_data['Recovered'].replace(np.nan, -1.0)
    merged_data['Active'] = merged_data['Active'].replace(np.nan, -1.0)
    merged_data['Incidence_Rate'] = merged_data['Incidence_Rate'].replace(np.nan, -1.0)
    merged_data['Case-Fatality_Ratio'] = merged_data['Case-Fatality_Ratio'].replace(np.nan, -1.0)
        
    merged_data_missing = merged_data.isnull().sum(axis = 0)
    # print("\n\n" ,merged_data.head(20).to_string() )
    return merged_data