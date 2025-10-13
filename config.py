local_crs = 'EPSG:32618'

dir_road_closure = 'data/nyc_street_flooding_20250218.csv'
dir_NYC_data_token = 'data/nyc_opendata_token.txt'

dir_city_boundary = 'data/nyc_boundary/nybb.shp'
dir_adapted_nyc_roads = 'data/nyc_roads_geometry_corrected.geojson'
dir_mapbox_token = './data/mapbox_token.txt'
dir_results = './results'
dir_cache_instance = './cache/instances'
dir_figures = './figures'

with open(dir_mapbox_token, "r") as file:
    mapbox_token = file.read()

# remove nodes to keep data when alignment, check hte __init__ of TrafficBayesNetwork for details
remove_data_from_nodes = ['4616195', '4616229', '4616223',]

sensor_count = 3

corr_thr = .6  # correlation threshold when building the bayesian network

weight_disruption = .5