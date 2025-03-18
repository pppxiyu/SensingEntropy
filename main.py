from config import *
import data as dd
import model as mo
import visualization as vis

# Geometry manual processing
road_data = dd.RoadData()
road_data.pull_nyc_dot_traffic(dir_NYC_data_token, ['2020-04-01', '2020-04-30'], True)
road_data.import_adapted_nyc_road(dir_adapted_nyc_roads)

# Identify flooding periods from closures
road_data.import_street_flooding(dir_road_closure, local_crs, 20)
# vis.overlap_roads_flood_plotly(
#     road_data.geo.to_crs(local_crs), road_data.closures.to_crs(local_crs), buffer=20,
#     mapbox_token=mapbox_token, save_dir=dir_figure_save
# )
# vis.bar_flood_prob(road_data.closure_p_per_segment.iloc[48])
# vis.map_flood_p(road_data.geo, road_data.closure_p_per_segment, mapbox_token, local_crs)
road_data.infer_flooding_time_citywide()
road_data.infer_flooding_time_per_road()

# Pull traffic data in flooding periods
road_data.pull_nyc_dot_traffic_flooding(dir_NYC_data_token)
road_data.resample_nyc_dot_traffic(road_data.speed)

# Fit marginal distributions
bayes_network = mo.TrafficBayesNetwork()
bayes_network.fit_speed(road_data.speed, max_components=3)
# for seg in road_data.speed['link_id'].unique()[0: 10]:
#     vis.dist_gmm_1d(
#         road_data.speed[road_data.speed['link_id'] == seg]['speed'].values,
#         bayes_network.gmm_per_segment[seg]
#     )
bayes_network.fit_flood(road_data.closures)

# Fit conditional (joint) distributions: road - road
bayes_network.build_network_from_geo(road_data.geo, remove_no_data_segment=True)
# vis.map_road_network_connections(road_data.geo, bayes_network.network, local_crs)
bayes_network.fit_joint_speed_n_speed(road_data.speed_resampled)
# for k, v in list(bayes_network.gmm_joint_road_road.items())[1: 10]:
#     if len(v[0]) == 1:
#         vis.dist_gmm_3d(v[1], k, v[0][0])

# Fit conditional (joint) distributions: flood - road
bayes_network.fit_joint_flood_n_speed(
    road_data.flood_time_per_road, road_data.speed_resampled, max_components=3,
)
# for k, v in bayes_network.gmm_joint_flood_road.items():
#     vis.dist_discrete_gmm(k, v)

# Measure unobserved network entropy
bayes_network.calculate_network_entropy()

# Measure observed network entropy (for one sensor)
# Update network first with observed distributions
for k, v in bayes_network.gmm_joint_flood_road.items():
    bayes_network.update_network_with_observed_dist(k, v['speed_no_flood'], max_components=3)
    bayes_network.update_network_with_observed_dist(k, v['speed_flood'], max_components=3)
# 0.01 * entropy {network updated with flood time dist} + 0.99 * entropy {network with non-flood time dist}
# Conditional entropy: H(A∣B)=∑_b P(B=b)H(A∣B=b)

# Place one sensor

# Place multiple sensors















pass

