from main import *

def data_cleaning_and_missing_values(location_data, test_data, train_data):
    # Case and train data
    train_and_test_methods(test_data)
    train_and_test_methods(train_data)
    # location data
    # location_methods(location_data)

def train_and_test_methods(dataset):
    replace_missing_with_default_cases(dataset)
    impute_dates(dataset)
    remove_missing_lat_and_long_cases(dataset)
    impute_age(dataset)
    impute_missing_countries_cases(dataset)
    impute_missing_provinces_case(dataset)
    convertDataToFloatsForCases(dataset)

def location_methods(location_data):
    impute_missing_provinces_loc(location_data)
    impute_missing_lat_long_loc(location_data)
    impute_missing_active_loc(location_data)
    impute_missing_caseFatal_loc(location_data)
    impute_missing_incidentRate_loc(location_data)
    convertLocDataToFloats(location_data)

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

########### LOCATION DATA METHODS ###########

def impute_missing_provinces_loc(location_data):
    # Impute missing province states for location_data
    # Lat long pair for certain province_states/places are unique hence we cannot guess province (replaced with unknown)
    temp = location_data.copy()
    geolocator = Nominatim(user_agent="http") 

    temp['Lat'] = temp.apply(lambda x: (geolocator.geocode(x['Country_Region'])).latitude if pd.isnull(x['Lat']) else int(x['Lat']), axis=1)
    temp['Long_'] = temp.apply(lambda x: (geolocator.geocode(x['Country_Region'])).longitude if pd.isnull(x['Long_']) else int(x['Long_']), axis=1)

    temp['Lat'] = temp['Lat'].astype(str)
    temp['Long_'] = temp['Long_'].astype(str)

    temp['lat-long'] = list(zip(temp.Lat, temp.Long_))

    g = temp.dropna(subset=['Province_State']).drop_duplicates('lat-long').set_index('lat-long')['Province_State']
    temp['Province_State'] = temp['Province_State'].fillna(temp['lat-long'].map(g))

    location_data['Province_State'] = temp['Province_State']

    location_data['Province_State'] = location_data['Province_State'].replace(np.nan, 'unknown')

def impute_missing_lat_long_loc(location_data):
    # impute lat and long for location data
    geolocator = Nominatim(user_agent="http") 
    h = location_data[(location_data['Lat'].isna()) & (location_data['Province_State'] != 'unknown')]
    h['Lat'] = h['Province_State'].apply(lambda x: (geolocator.geocode(x).latitude))
    h['Long_'] = h['Province_State'].apply(lambda x: (geolocator.geocode(x).longitude))
    for row in h.index:
        location_data.loc[row, 'Lat'] = h.loc[row, 'Lat']
        location_data.loc[row, 'Long_'] = h.loc[row, 'Long_']

    h = location_data[(location_data['Lat'].isna())]
    h['Lat'] = h['Country_Region'].apply(lambda x: (geolocator.geocode(x).latitude))
    h['Long_'] = h['Country_Region'].apply(lambda x: (geolocator.geocode(x).longitude))
    for row in h.index:
        location_data.loc[row, 'Lat'] = h.loc[row, 'Lat']
        location_data.loc[row, 'Long_'] = h.loc[row, 'Long_']
    
def impute_missing_active_loc(location_data):
    # Impute missing Active cases for location_data
    # Active = confirmed - deaths - recovered -- went negative so set to 0
    location_data['Active'] = location_data['Active'].replace(np.nan, '0')


def impute_missing_caseFatal_loc(location_data):
    # Fatality ratio = (deaths / confirmed) x 100
    location_data['Confirmed'] = location_data['Confirmed'].astype(float)
    location_data['Deaths'] = location_data['Deaths'].astype(float)
    location_data.loc[location_data['Case-Fatality_Ratio'].isna(), 'Case-Fatality_Ratio'] = (location_data['Deaths']/location_data['Confirmed']) * 100
    location_data['Case-Fatality_Ratio'] = location_data['Case-Fatality_Ratio'].replace(np.nan, 0.0)

def impute_missing_incidentRate_loc(location_data):
    # make mapping of combined key to each incident rate 
    # make different mapping for mean and combined key 
    # assign
    temp = location_data.copy()
    temp['Province_State'] = temp['Province_State'].str.strip()
    temp['Country_Region'] = temp['Country_Region'].str.strip()
    temp['Combined_Key'] = temp['Province_State'] + ', ' + temp['Country_Region'] 

    h = temp.groupby('Combined_Key', as_index=False)['Incidence_Rate'].mean()
    g = h.dropna(subset=['Incidence_Rate']).drop_duplicates('Combined_Key').set_index('Combined_Key')['Incidence_Rate']
    temp['Incidence_Rate'] = temp['Incidence_Rate'].fillna(temp['Combined_Key'].map(g))

    temp['Confirmed'] = temp['Confirmed'].astype(float)
    temp['Incidence_Rate'] = temp['Incidence_Rate'].astype(float)
    temp['Incidence_Rate'] = temp.apply(lambda x: x['Confirmed'] / 100000.0  if pd.isnull(x['Incidence_Rate']) else x['Incidence_Rate'], axis=1)

    location_data['Incidence_Rate'] = temp['Incidence_Rate']
    location_data['Combined_Key'] = temp['Combined_Key']

#converting location data values
def convertLocDataToFloats(location_data):
    location_data['Lat'] = location_data['Lat'].astype(float)
    location_data['Long_'] = location_data['Long_'].astype(float)
    location_data['Confirmed'] = location_data['Confirmed'].astype(float)
    location_data['Deaths'] = location_data['Deaths'].astype(float)
    location_data['Recovered'] = location_data['Recovered'].astype(float)
    location_data['Active'] = location_data['Active'].astype(float)
    location_data['Incidence_Rate'] = location_data['Incidence_Rate'].astype(float)
    location_data['Case-Fatality_Ratio'] = location_data['Case-Fatality_Ratio'].astype(float)
