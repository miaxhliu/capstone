#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip install geopy')
get_ipython().system('pip install tqdm')
get_ipython().system('pip install BeautifulSoup4')
get_ipython().system('conda install -c conda-forge folium=0.5.0 --yes')

import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import json
from geopy.geocoders import Nominatim
from tqdm import tqdm
from geopy.extra.rate_limiter import RateLimiter
from math import sqrt, pi
import requests
from pandas.io.json import json_normalize
import matplotlib.cm as cm

from matplotlib import pyplot as plt
import matplotlib.colors as colors
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN 
from sklearn.cluster import KMeans 
import folium

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


get_ipython().system('conda install --yes seaborn')


# In[3]:


get_ipython().system('conda install --yes html5lib')


# In[3]:


# Importing data from Wikipedia
df_planning_areas = pd.read_html("http://en.wikipedia.org/wiki/Planning_Areas_of_Singapore", flavor='html5lib', header=0)[2]
df_planning_areas.head()


# In[4]:


# Dropping not-needed columns, renaming columns, and replacing empty values
df_planning_areas.drop(columns=["Malay", "Chinese", "Pinyin", "Tamil"], inplace = True)
df_planning_areas.columns = ["Planning Area", "Region", "Area", "Population", "Density"]
df_planning_areas.replace("*", 0, inplace=True)
df_planning_areas.head()


# In[5]:


df_planning_areas.dtypes


# In[6]:


# Correcting data types
df_planning_areas = df_planning_areas.astype({"Population":"float64", "Density":"float64"})
df_planning_areas.dtypes


# In[7]:


# Initialising geocoding agent
geolocator = Nominatim(user_agent="Mozilla/76.0")
location = geolocator.geocode("Singapore")
latitude = location.latitude
longitude = location.longitude
print(f"Coordinates of Singapore are {latitude}, {longitude}")


# In[8]:



# Getting coordinates of each Planning Area, and adding suffix to search query
tqdm.pandas()
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
coords = (df_planning_areas["Planning Area"] + " suburb, Singapore").progress_apply(geocode)


# In[10]:


# Adding two new empty columns to dataframe
df_planning_areas["Latitude"] = np.nan
df_planning_areas["Longitude"] = np.nan
df_planning_areas.head()


# In[9]:


print(coords)


# In[14]:


# Populating the Latitude and Longitude columns with data from coords
for index in df_planning_areas.index:
    if coords[index]:
       df_planning_areas.at[index, 'Latitude'] = coords[index].latitude
       df_planning_areas.at[index, 'Longitude'] = coords[index].longitude
    else:
       df_planning_areas["Latitude"] = np.nan
       df_planning_areas["Longitude"] = np.nan
df_planning_areas


# In[16]:


df_planning_areas = np.delete(df_planning_areas, np.where(df_planning_areas.Latitude == np.nan))


# In[19]:


df_planning_areas = df_planning_areas.dropna(how='any',axis=0)


# In[20]:


df_planning_areas


# In[22]:


# Adding a new Search Radius column into dataframe, and re-ordering columns
# The new Search Radius will be used in the Foursquare API search query
df_planning_areas["Search Radius"] = df_planning_areas["Area"].apply(lambda x: round(sqrt(x/pi)*1000))
df_planning_areas = df_planning_areas[['Planning Area', 'Region', 'Area', 'Search Radius', 'Population', 'Density', 'Latitude', 'Longitude']]


# In[23]:


# Visualising map using folium
# for loop used to add map markers for each Planning Area
map_singapore = folium.Map(location = [latitude, longitude], zoom_start = 12)
for lat, lng, region, name in zip(df_planning_areas['Latitude'], df_planning_areas['Longitude'], df_planning_areas['Region'], df_planning_areas['Planning Area']):
    label = '{}, {}'.format(name, region)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(map_singapore)

map_singapore


# In[28]:


# By running the geocoder again with the additional arugument exactly_one=False, 2 results are returned
boon_lay = geolocator.geocode(query="Singapore River, Singapore", exactly_one=False)
boon_lay


# In[29]:


# Determining which coordinates are currently stored in Boon Lay
df_planning_areas[df_planning_areas['Planning Area'] == 'Singapore River']


# In[30]:



# Replacing the new Boon Lay coordinates into the dataframe
df_planning_areas.at[3, 'Latitude'] = boon_lay[1].latitude
df_planning_areas.at[3, 'Longitude'] = boon_lay[1].longitude
df_planning_areas[df_planning_areas['Planning Area'] == 'Singapore River']


# In[27]:


# Visualising the map to verify changes were successfuly made
map_singapore = folium.Map(location = [df_planning_areas.loc[3, 'Latitude'], 
                                       df_planning_areas.loc[3, 'Longitude']], 
                            zoom_start = 13)
for lat, lng, region, name in zip(df_planning_areas['Latitude'], 
                                  df_planning_areas['Longitude'], 
                                  df_planning_areas['Region'], 
                                  df_planning_areas['Planning Area']):
    label = '{}, {}'.format(name, region)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(map_singapore)

map_singapore


# In[35]:


# Foursquare developer API credentials
CLIENT_ID = 'T1LUMOHYFVR21KSZSVKTKIFO1DPONRPOL4PGKBLWSYTPNDNV'
CLIENT_SECRET = 'JBKZMRN5WS2PBXLZDUMMLCYJIDZELUNSSP3QKM51GSHXXWB0' 
VERSION = '20180604'
LIMIT = 100


# In[36]:


# Defining a function to get nearby venues, using the Foursquare API, and extracting relevant information from the JSON response
# The function returns nearby venues in that Planning Area
def getNearbyVenues(names, latitudes, longitudes, radius):
    
    venues_list=[]
    for name, lat, lng, radius in zip(names, latitudes, longitudes, radius):
        print(name)
            
        # create the API request URL
        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            radius, 
            LIMIT)
            
        # make the GET request
        results = requests.get(url).json()["response"]['groups'][0]['items']
        
        # return only relevant information for each nearby venue
        venues_list.append([(
            name, 
            lat, 
            lng, 
            v['venue']['name'], 
             v['venue']['location']['lat'], 
            v['venue']['location']['lng'],  
            v['venue']['categories'][0]['name']) for v in results])

    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['Planning Area', 
                  'PA Latitude', 
                  'PA Longitude', 
                  'Venue', 
                  'Venue Latitude', 
                  'Venue Longitude', 
                  'Venue Category']
    
    return(nearby_venues)


# In[43]:



# Calling the user-defined function, and saving the results into a variable
if getNearbyVenues(names=df_planning_areas['Planning Area'],
                                   latitudes=df_planning_areas['Latitude'],
                                   longitudes=df_planning_areas['Longitude'],
                                   radius=df_planning_areas['Search Radius']
                                  ):
        singapore_venues = getNearbyVenues(names=df_planning_areas['Planning Area'],
                                   latitudes=df_planning_areas['Latitude'],
                                   longitudes=df_planning_areas['Longitude'],
                                   radius=df_planning_areas['Search Radius']
                                  )


# In[44]:


print(singapore_venues.shape)
singapore_venues.head()


# In[39]:


print("Number of unique venue categories in Singapore planning areas:{}".format(len(singapore_venues["Venue Category"].unique())))


# In[ ]:




