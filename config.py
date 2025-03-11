
local_crs = 'EPSG:32618'
dir_road_closure = './data/Street_Flooding_20250218.csv'
dir_NYC_data_token = './data/NYC_token.txt'
dir_mapbox_token = './data/mapbox_token.txt'
dir_figure_save = './data/figures'
dir_adapted_nyc_roads = './data/nyc_roads_adapted.geojson'

with open(dir_mapbox_token, "r") as file:
    mapbox_token = file.read()

