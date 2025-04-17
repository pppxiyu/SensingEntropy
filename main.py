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
# vis.map_roads_n_flood_plt(
#     road_data.geo.to_crs(local_crs), road_data.closures.to_crs(local_crs), buffer=20,
# )
# vis.map_flood_p(road_data.geo, road_data.closure_p_per_segment, mapbox_token, local_crs)
road_data.infer_flooding_time_citywide()
road_data.infer_flooding_time_per_road()

# Pull traffic data in flooding periods
road_data.pull_nyc_dot_traffic_flooding(dir_NYC_data_token)
road_data.resample_nyc_dot_traffic(road_data.speed)

# Fit marginals
bayes_network = mo.TrafficBayesNetwork()
bayes_network.fit_speed(road_data.speed, max_components=3)
# for seg in road_data.speed['link_id'].unique()[0: 10]:
#     vis.dist_histo_gmm_1d(
#         road_data.speed[road_data.speed['link_id'] == seg]['speed'].values,
#         bayes_network.gmm_per_segment[seg]
#     )
bayes_network.fit_flood(road_data.closures)
# vis.bar_flood_prob(bayes_network.closure_p_per_segment.iloc[48])  # FIGURE 4: flood probability

# Fit joints: road - road
bayes_network.build_network_from_geo(road_data.geo, remove_no_data_segment=True)
# vis.map_roads_n_topology_plt(
#     geo_roads=road_data.geo.copy(),
#     network=bayes_network.network.copy(), network_geo_roads=road_data.geo.copy(),
#     local_crs=local_crs
# ) # FIGURE 1: road network and topology
bayes_network.fit_joint_speed_n_speed(road_data.speed_resampled)
# for k, v in list(bayes_network.gmm_joint_road_road.items()):
#     if len(v[0]) == 1:  # the vis only support 3d
#         vis.dist_gmm_3d(v[1], k, v[0][0])  # FIGURE 2: joint distributions between roads

# Fit relationships flood - road
bayes_network.fit_joint_flood_n_speed(
    road_data.flood_time_per_road, road_data.speed_resampled, max_components=3,
)
# for k, v in bayes_network.gmm_joint_flood_road.items():
#     vis.dist_discrete_gmm(v, bayes_network.gmm_per_segment[k],)  # FIGURE 3: distribution with observation

# Measure unobserved network entropy
bayes_network.calculate_network_entropy()
print()

# Measure observed network entropy (for one sensor)
# Update network first with observed distributions
for k, v in bayes_network.gmm_joint_flood_road.items():
    if (v['speed_no_flood'] is not None) and (v['speed_flood'] is not None):
        marginals_no_flood, joints_no_flood = bayes_network.update_network_with_observed_dist(
            k, v['speed_no_flood'], bayes_network.gmm_per_segment, bayes_network.gmm_joint_road_road, verbose=0
        )
        marginals_flood, joints_flood = bayes_network.update_network_with_observed_dist(
            k, v['speed_flood'], bayes_network.gmm_per_segment, bayes_network.gmm_joint_road_road, verbose=0
        )
        bayes_network.calculate_network_conditional_entropy([
            {'p': 1 - v['p_flood'], 'marginals': marginals_no_flood, 'joints': joints_no_flood},
            {'p': v['p_flood'], 'marginals': marginals_flood, 'joints': joints_flood},
        ])

# Place one sensor

# Place multiple sensors















pass

