local_crs = 'EPSG:32618'
dir_road_closure = './data/Street_Flooding_20250218.csv'
dir_NYC_data_token = './data/NYC_token.txt'
dir_mapbox_token = './data/mapbox_token.txt'
dir_figure_save = './data/figures'
dir_adapted_nyc_roads = './data/nyc_roads_adapted.geojson'
dir_city_boundary = './data/nybb_25a/nybb.shp'

with open(dir_mapbox_token, "r") as file:
    mapbox_token = file.read()

# remove nodes to keep data when alignment, check hte __init__ of TrafficBayesNetwork for detals
remove_data_from_nodes = ['4616195', '4616229', '4616223',]

sensor_count = 1

