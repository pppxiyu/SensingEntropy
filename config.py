local_crs = 'EPSG:32618'

dir_road_closure = 'data/nyc_street_flooding_20250218.csv'
dir_NYC_data_token = 'data/nyc_opendata_token.txt'
dir_city_boundary = 'data/nyc_boundary/nybb.shp'
dir_adapted_nyc_roads = 'data/nyc_roads_geometry_corrected.geojson'

dir_results = './results'
dir_cache_instance = './cache/instances'
dir_figures = './figures'

# remove nodes to keep data when alignment, check the __init__ of TrafficBayesNetwork for details
remove_data_from_nodes = ['4616195', '4616229', '4616223',]

sensor_count = 9

corr_thr = .6  # correlation threshold when building the bayesian network
weight_thr = 0.2  # thr used when building the flood BN

thr_flood, thr_not_flood = 1, 1  # threshold of p to determine flood / not flood state when inferring

weight_disruption = 0.5  # the weight of the objective Disruption

prune_rate = 0.01  # used to multiply the min value of the flood p for pruning the scenario tree