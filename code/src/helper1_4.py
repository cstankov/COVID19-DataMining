
def transform_location_data(location_data):
    temp = location_data.copy()

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
    temp2['Incidence_Rate'] = (temp2['Confirmed'] / 100000)

    # calculate fat ratio = deaths/confirmed * 100
    temp2['Case-Fatality_Ratio'] = (temp2['Deaths'] / temp2['Confirmed']) * 100

    location_data = temp2.copy()