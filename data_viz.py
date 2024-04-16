from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import os


latitudes = []
longitudes = []

for filename in os.listdir('data/street'):
    if not filename.endswith("_street.png") or filename.startswith('.'):
            continue
    
    lat_lng = filename.split('_street.png')[0]
    lat, lng = lat_lng.split(',')
    latitudes.append(float(lat))
    longitudes.append(float(lng))

fig, ax = plt.subplots()

m = Basemap(projection='merc', llcrnrlat=24, urcrnrlat=50, llcrnrlon=-125, urcrnrlon=-66, resolution='i', ax=ax)
m.bluemarble(scale=1)
m.drawcoastlines()
m.drawstates()
m.drawcountries()

x, y = m(longitudes, latitudes)

m.scatter(x, y, color='red', marker='o', s=10) 

plt.show()