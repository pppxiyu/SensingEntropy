from config import *
import data as dd
import model as mo
import visualization as vis

import random
import numpy as np
random.seed(0)
np.random.seed(0)

"""
Data
"""

# Geometry manual processing
road_data = dd.RoadData()
road_data.pull_nyc_dot_traffic(dir_NYC_data_token, ['2020-04-01', '2020-04-30'], True)
road_data.import_adapted_nyc_road(dir_adapted_nyc_roads)

# Identify flooding periods from df
road_data.import_street_flooding(dir_road_closure, local_crs, 20)
# vis.map_roads_n_flood_plt(
#     road_data.geo.to_crs(local_crs), road_data.df.to_crs(local_crs), buffer=20,
# )  # FIGURE X: flood point and roads
# vis.map_flood_p(road_data.geo, road_data.closure_p_per_segment, mapbox_token, local_crs)  # FIGURE X: flood p
road_data.infer_flooding_time_citywide()
road_data.infer_flooding_time_per_road()

# Pull traffic data in flooding periods
road_data.pull_nyc_dot_traffic_flooding(dir_NYC_data_token)
road_data.resample_nyc_dot_traffic(road_data.speed)

"""
Modeling
"""

# Fit marginals - flood
bayes_network_f = mo.FloodBayesNetwork()
bayes_network_f.fit_marginal(road_data.closures)
# vis.bar_flood_prob(bayes_network_f.marginals.iloc[48])  # FIGURE 5: flood probability

# Fit conditionals - flood
bayes_network_f.build_network_by_co_occurrence(road_data.closures,  weight_thr=0.2)
bayes_network_f.fit_conditional(road_data.closures)
bayes_network_f.build_bayes_network()

# Fit marginals - speed
bayes_network_t = mo.TrafficBayesNetwork(10000, 3)
bayes_network_t.fit_marginal(road_data.speed, )
# for seg in road_data.speed['link_id'].unique()[0: 10]:
#     vis.dist_histo_gmm_1d(
#         road_data.speed[road_data.speed['link_id'] == seg]['speed'].values,
#         bayes_network_t.gmm_per_road[seg]
#     )  # FIGURE X: marginal distributions on roads

# Fit joints - speed
bayes_network_t.build_network_from_geo(road_data.geo, mode='causal', remove_no_data_segment=True)
# vis.map_roads_n_topology_plt(
#     geo_roads=road_data.geo.copy(),
#     network=bayes_network_t.network.copy(), network_geo_roads=road_data.geo.copy(),
#     local_crs=local_crs, city_shp_path=dir_city_boundary,
# ) # FIGURE 1: road network and topology
bayes_network_t.fit_joint(road_data.speed_resampled)
# for k, v in list(bayes_network_t.gmm_joint_road_road.items()):
#     if len(v['parents']) == 1:  # the vis only support 3d
#         print(f'Joint distributions at {k}')
#         vis.dist_gmm_3d(v['joints'], k, v['parents'][0])  # FIGURE 2: joint distributions between roads

# Fit signals
bayes_network_t.fit_joint_flood_n_speed(
    road_data.flood_time_per_road, road_data.speed_resampled, bayes_network_f.marginals
)
# for k, v in bayes_network_t.gmm_joint_flood_road.items():
#     print(f'Distributions under observations at {k}')
#     vis.dist_discrete_gmm(v, bayes_network_t.gmm_per_road[k],)  # FIGURE 3: distribution with observation

"""
Inference and placement
"""

# Measure unobserved network entropy
entropy_original = bayes_network_t.calculate_network_entropy()
print()

# Update network with observation; Measure long-term observed network entropy
entropies = {}
for k, v in bayes_network_t.gmm_joint_flood_road.items():
    if (v['speed_no_flood'] is not None) and (v['speed_flood'] is not None):
        inferred_signals_no_flood = bayes_network_t.convert_state_to_dist(
            bayes_network_f.infer_node_states(k, 0, 1, 1)
        )
        inferred_signals_flood = bayes_network_t.convert_state_to_dist(
            bayes_network_f.infer_node_states(k, 1, 1, 1)
        )
        # # if len({
        # #     kk: vv for kk, vv in {**inferred_signals_no_flood, **inferred_signals_flood}.items()
        # #     if vv is not None
        # # }) > 0:
        # inferred_signals_no_flood, inferred_signals_flood = {}, {}
        # print(inferred_signals_no_flood, inferred_signals_flood)
        marginals_no_flood, joints_no_flood = bayes_network_t.update_network_with_multiple_soft_evidence_downward(
            {**{k: v['speed_no_flood']}, **inferred_signals_no_flood},
            bayes_network_t.gmm_per_road, bayes_network_t.gmm_joint_road_road, verbose=0
        )
        marginals_no_flood, joints_no_flood = bayes_network_t.update_network_with_multiple_soft_evidence_upward(
            {**{k: v['speed_no_flood']}, **inferred_signals_no_flood},
            marginals_no_flood, joints_no_flood, verbose=0,
        )
        marginals_flood, joints_flood = bayes_network_t.update_network_with_multiple_soft_evidence_downward(
            {**{k: v['speed_flood']}, **inferred_signals_flood},
            bayes_network_t.gmm_per_road, bayes_network_t.gmm_joint_road_road, verbose=0
        )
        marginals_flood, joints_flood = bayes_network_t.update_network_with_multiple_soft_evidence_upward(
            {**{k: v['speed_flood']}, **inferred_signals_flood},
            marginals_flood, joints_flood, verbose=0,
        )
        entropies[k] = bayes_network_t.calculate_network_conditional_entropy([
            {'p': 1 - v['p_flood'], 'marginals': marginals_no_flood, 'joints': joints_no_flood},
            {'p': v['p_flood'], 'marginals': marginals_flood, 'joints': joints_flood},
        ])

        # marginals_no_flood, joints_no_flood = bayes_network_t.update_network_with_soft_evidence(
        #     k, v['speed_no_flood'], bayes_network_t.gmm_per_road, bayes_network_t.gmm_joint_road_road, verbose=0
        # )
        # marginals_flood, joints_flood = bayes_network_t.update_network_with_soft_evidence(
        #     k, v['speed_flood'], bayes_network_t.gmm_per_road, bayes_network_t.gmm_joint_road_road, verbose=0
        # )
        # entropies[k] = bayes_network_t.calculate_network_conditional_entropy([
        #     {'p': 1 - v['p_flood'], 'marginals': marginals_no_flood, 'joints': joints_no_flood},
        #     {'p': v['p_flood'], 'marginals': marginals_flood, 'joints': joints_flood},
        # ])


        # if (k in joints_flood.keys()) and (len(joints_flood[k][0]) == 1):
        #     z_max = vis.dist_gmm_3d(joints_flood[k][1], k, joints_flood[k][0][0], return_z_max=True)
        #     vis.dist_gmm_3d(
        #         bayes_network_t.gmm_joint_road_road[k][1], k, bayes_network_t.gmm_joint_road_road[k][0][0],
        #         z_limit=z_max
        #     )  # FIGURE 4: joint distributions changes

"""
A function is needed here to combine the two posteriors by the selected sensor into one prior for iteration.
"""

# vis.map_roads_w_values(
#     road_data.geo.copy(), entropies, city_shp_path=dir_city_boundary
# )  # FIGURE 6: entropy map

pass

