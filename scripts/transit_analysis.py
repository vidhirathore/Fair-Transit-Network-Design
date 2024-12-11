import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
import geopandas as gpd
import folium
from folium import plugins

if not os.path.exists('../visuals'):
    os.makedirs('../visuals')

data_path = '../data/'
cities = pd.read_csv(os.path.join(data_path, "cities.csv"))
stations = pd.read_csv(os.path.join(data_path, "stations.csv"))
tracks = pd.read_csv(os.path.join(data_path, "tracks.csv"))
lines = pd.read_csv(os.path.join(data_path, "lines.csv"))
track_lines = pd.read_csv(os.path.join(data_path, "track_lines.csv"))

print("Loaded data successfully.")

stations = stations.dropna(subset=['closure', 'name', 'opening'])
stations = stations[stations.closure >= 9999]
stations = stations[stations.opening > 0]
stations = stations[stations.opening <= 2030]
stations.columns = ['id', 'stations_name', 'geometry', 'buildstart', 'opening', 'closure', 'city_id']
stations['Long'] = stations['geometry'].apply(lambda x: float(x.split('POINT(')[1].split(' ')[0]))
stations['Lat'] = stations['geometry'].apply(lambda x: float(x.split('POINT(')[1].split(' ')[1].split(')')[0]))
id_country = pd.DataFrame({'city_id': cities.id, 'country': cities.country, 'name': cities.name})

stations = pd.merge(stations, id_country)
stations.head()

osaka_lines = lines[lines.city_id == 91]
osaka_track_lines = track_lines[track_lines.city_id == 91]
osaka_tracks = tracks[tracks.city_id == 91].drop(columns=['buildstart', 'opening', 'closure', 'city_id'])
osaka_tracks.columns = ['section_id', 'geometry', 'length']
osaka_track_lines = pd.merge(osaka_track_lines, osaka_tracks)
osaka_track_lines = osaka_track_lines.drop(columns=['id', 'created_at', 'updated_at', 'city_id'])
osaka_track_lines.columns = ['section_id', 'id', 'geometry', 'length']
osaka_lines = pd.merge(osaka_track_lines, osaka_lines)
osaka_stations = stations[stations['city_id'] == 91]

x, y, z = [], [], []
for i in range(len(osaka_lines)):
    sp = osaka_lines.iloc[i].geometry.split('(')[1].split(')')[0].split(',')
    for point in sp:
        lon, lat = map(float, point.strip().split(' '))
        x.append(lon)
        y.append(lat)
        z.append(osaka_lines.url_name.iloc[i])

fix = pd.DataFrame({'x': x, 'y': y, 'z': z})

plt.figure(figsize=(12, 8))
sns.scatterplot(x="x", y="y", hue="z", data=fix, palette='tab10', legend='full')
plt.title("Lines for Osaka", size=20)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend(title='Line Name', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('../visuals/osaka_lines_scatter.png')
plt.close()

plt.figure(figsize=(10, 8))
(top_tracks := (osaka_lines.groupby('url_name')['length'].sum() / 1000)
              .sort_values(ascending=False)[:10]
              .sort_values()).plot.barh(color='skyblue')
plt.xlabel('Length (km)')
plt.title("Top 10 Tracks by Length in Osaka", size=20)
plt.tight_layout()
plt.savefig('../visuals/osaka_top_tracks.png')
plt.close()

plt.figure(figsize=(12, 8))
(osaka_stations.groupby('opening')['id'].count()
 .plot(kind='line', marker='o', color='green'))
plt.xlabel('Year')
plt.ylabel('Number of Stations')
plt.title("Number of Opening Stations by Year in Osaka", size=20)
plt.tight_layout()
plt.savefig('../visuals/osaka_opening_stations.png')
plt.close()

plt.figure(figsize=(10, 8))
(osaka_lines['name'].value_counts()[:10]
 .sort_values()
 .plot.barh(color='coral'))
plt.xlabel('Counts')
plt.title("Top 10 Lines by Number of Stations in Osaka", size=20)
plt.tight_layout()
plt.savefig('../visuals/osaka_top_lines.png')
plt.close()

stations_osaka = osaka_stations.head(2000)
osaka_map = folium.Map(location=[34.53, 135.5], zoom_start=12)

osaka_stations_map = plugins.MarkerCluster().add_to(osaka_map)
for _, row in stations_osaka.iterrows():
    folium.Marker(
        location=[row['Lat'], row['Long']],
        popup=row['stations_name']
    ).add_to(osaka_stations_map)

osaka_map.save('../visuals/osaka_stations_map.html')

print("All visualizations have been saved in the 'visuals/' directory.")
