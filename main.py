from config import *
import data as dd
import model as mo
import visualization as vis

# Geometry manually processing
road_data = dd.RoadData()
road_data.pull_nyc_dot_traffic(dir_NYC_data_token, ['2020-04-01', '2020-04-30'], True)

# Identify flooding days from closures
road_data.import_street_flooding(dir_road_closure, local_crs, 20)
# vis.overlap_roads_flood_plotly(
#     road_data.road_geo.to_crs(local_crs), road_data.road_closure.to_crs(local_crs), buffer=20,
#     mapbox_token=mapbox_token, save_dir=dir_figure_save
# )
road_data.convert_closure_2_prob()
# vis.map_flood_p(road_data.road_geo, road_data.road_closure_p, mapbox_token, local_crs)

# Pull traffic data in flooding periods
road_data.pull_nyc_dot_traffic(dir_NYC_data_token, ['2020-04-01', '2020-04-30'], False)
road_data.resample_nyc_dot_traffic(road_data.road_speed)

# Fit conditional distributions of traffic speeds
bayes_network = mo.TrafficBayesNetwork()
bayes_network.fit_gmm_4_segment(road_data.road_speed, max_components=3)
# for seg in road_data.road_speed['link_id'].unique()[0: 10]:
#     vis.dist_gmm_1d(
#         road_data.road_speed[road_data.road_speed['link_id'] == seg]['speed'].values,
#         bayes_network.gmm_per_segment[seg]
#     )
road_data.import_adapted_nyc_road(dir_adapted_nyc_roads)
bayes_network.build_network_from_geo(road_data.road_geo)
# vis.map_road_network_connections(road_data.road_geo, bayes_network.network, local_crs)
bayes_network.fit_gmm_4_dependency(road_data.road_speed_resampled)
# for k, v in bayes_network.gmm_dependency.items():
#     if len(v[0]) == 1:
#         vis.dist_gmm_3d(v[1], k, v[0][0])

# Fit conditional distributions between closure and traffic speeds


# Remove flooding node and calculate global join distribution


# Greedy algorithm





pass

