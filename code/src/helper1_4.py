from main import *


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


def transform_location_data(location_data):

    convertLocDataToFloats(location_data)
    location_data['Province_State'] = location_data['Province_State'].fillna('unknown')
    
    temp = location_data.copy()

    temp['Country_Region'] = temp.apply(lambda x: 'United States'if x['Country_Region'] == 'US' else x['Country_Region'], axis = 1)

    #['province'] = dataset.apply(lambda x: "Puerto Rico" if x['country'] == "Puerto Rico" else x['province'], axis=1 )
    temp['Country_Region'] = temp['Country_Region'].str.strip()
    temp['Province_State'] = temp['Province_State'].str.strip()
    
    # temp.loc[(temp['Country_Region'] == 'US'), 'Country_Region'] = 'United States'
    temp['Combined_Key'] = temp['Province_State'] + ', ' + temp['Country_Region']

    lat = temp.groupby('Combined_Key', as_index=False)['Lat'].mean()
    long = temp.groupby('Combined_Key', as_index=False)['Long_'].mean()
    conf = temp.groupby('Combined_Key', as_index=False)['Confirmed'].sum()
    death = temp.groupby('Combined_Key', as_index=False)['Deaths'].sum()
    recov = temp.groupby('Combined_Key', as_index=False)['Recovered'].sum()

    temp2 = temp.copy()

    temp2.drop_duplicates(subset=['Combined_Key'], keep="last", inplace=True)
    temp2.reset_index(drop=True, inplace=True)

    temp2['Lat'] = lat['Lat']
    temp2['Long_'] = long['Long_']
    temp2['Confirmed'] = conf['Confirmed']
    temp2['Deaths'] = death['Deaths']
    temp2['Recovered'] = recov['Recovered']

    temp2['Confirmed'] = temp2['Confirmed'].astype(float)
    temp2['Deaths'] = temp2['Deaths'].astype(float)
    temp2['Recovered'] = temp2['Recovered'].astype(float)

    # calculate active cases = confirmed - deaths - recovered 
    temp2['Active'] = (temp2['Confirmed'] - temp2['Deaths'] - temp2['Recovered'])

    # Calculate incidence rate = confirmed / 100000
    temp2['Incidence_Rate'] = (temp2['Confirmed'] / 100000.0)

    # calculate fat ratio = deaths/confirmed * 100
    temp2['Case-Fatality_Ratio'] = (temp2['Deaths'] / temp2['Confirmed']) * 100
    location_data = temp2.copy()

    return location_data