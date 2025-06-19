import data
import model
from config import *
import data as dd
import model as mo
import visualization as vis

import random
import numpy as np
random.seed(0)
np.random.seed(0)

import pickle
import os
import warnings

"""
Load cache
"""
road_data = data.RoadData()
load_r = road_data.load_instance('./cache/classes/road_data')
bayes_network_f = model.FloodBayesNetwork()
load_f = bayes_network_f.load_instance('./cache/classes/bayes_network_f')
bayes_network_t = model.TrafficBayesNetwork()
load_t = bayes_network_t.load_instance('./cache/classes/bayes_network_t')


if not load_r:
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

    road_data.save_instance('./cache/classes/road_data')

if not load_f:
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

    bayes_network_f.save_instance('./cache/classes/bayes_network_f')

if not load_t:
    # Init
    bayes_network_t = mo.TrafficBayesNetwork(
        speed=road_data.speed_resampled, road_geo=road_data.geo, network_mode='causal',
        n_samples=10000, n_components=12, fitting_mode='one-off',
        remove_nodes=remove_data_from_nodes,
    )
    # vis.map_roads_n_topology_plt(
    #     geo_roads=road_data.geo.copy(),
    #     network=bayes_network_t.network.copy(), network_geo_roads=road_data.geo.copy(),
    #     local_crs=local_crs, city_shp_path=dir_city_boundary,
    # ) # FIGURE 1: road network and topology

    # Fit joints - speed
    bayes_network_t.fit_joint()
    # for k, v in list(bayes_network_t.joints.items()):
    #     if len(v['parents']) == 1:  # the vis only support 3d
    #         print(f'Joint distributions at {k}')
    #         vis.dist_gmm_3d(v['joints'], k, v['parents'][0])  # FIGURE 2: joint distributions between roads

    # Fit marginals - speed
    bayes_network_t.fit_marginal_from_data()
    # for seg in road_data.speed['link_id'].unique()[0: 10]:
    #     vis.dist_histo_gmm_1d(
    #         road_data.speed[road_data.speed['link_id'] == seg]['speed'].values,
    #         bayes_network_t.marginals[seg]
    #     )  # FIGURE X: marginal distributions on roads

    # Fit signals
    bayes_network_t.fit_signal(road_data.flood_time_per_road, bayes_network_f.marginals)
    # for k, v in bayes_network_t.signal.items():
    #     print(f'Distributions under observations at {k}')
    #     vis.dist_discrete_gmm(v, bayes_network_t.marginals[k],)  # FIGURE 3: distribution with observation

    bayes_network_t.organize_keys()
    bayes_network_t.save_instance('./cache/classes/bayes_network_t')
# mo.check_gmr_bn_consistency(list(bayes_network_t.marginals.keys()), bayes_network_t.joints)

"""
Inference and placement
"""
# Update network with observation
marginals = bayes_network_t.marginals.copy()
joints = bayes_network_t.joints.copy()
for n in range(sensor_count):

    # get records
    try:
        with open(f"./cache/results/sensing_{n}.pkl", "rb") as f:
            results = pickle.load(f)
            print(f"Loaded sensing_{n} from disk.")
    except (FileNotFoundError, pickle.UnpicklingError, EOFError) as e:
        print(f"Loading failed ({e}), continuing with calculation.")

        results = {}
        for k, v in bayes_network_t.signal.items():
            inferred_signals_no_flood = bayes_network_t.convert_state_to_dist(
                bayes_network_f.infer_node_states(k, 0, 1, 1)
            )
            inferred_signals_flood = bayes_network_t.convert_state_to_dist(
                bayes_network_f.infer_node_states(k, 1, 1, 1)
            )
            marginals_no_flood, joints_no_flood = bayes_network_t.update_network_with_multiple_soft_evidence(
                {**{k: v['speed_no_flood']}, **inferred_signals_no_flood},
                marginals, joints, verbose=0
            )
            marginals_flood, joints_flood = bayes_network_t.update_network_with_multiple_soft_evidence(
                {**{k: v['speed_flood']}, **inferred_signals_flood},
                marginals, joints, verbose=0
            )
            bn_updated = [
                {'p': 1 - v['p_flood'], 'marginals': marginals_no_flood, 'joints': joints_no_flood},
                {'p': v['p_flood'], 'marginals': marginals_flood, 'joints': joints_flood},
            ]
            results[k] = {
                'bn_updated': bn_updated,
                # 'entropy': bayes_network_t.calculate_network_conditional_entropy(bn_updated, label=k),
                'info_gain': bayes_network_t.calculate_network_conditional_kl_divergence(bn_updated, label=k),
            }
            # if (k in joints_flood.keys()) and (len(joints_flood[k][0]) == 1):
            #     z_max = vis.dist_gmm_3d(joints_flood[k][1], k, joints_flood[k][0][0], return_z_max=True)
            #     vis.dist_gmm_3d(
            #         bayes_network_t.joints[k][1], k, bayes_network_t.joints[k][0][0],
            #         z_limit=z_max
            #     )  # FIGURE 4: joint distributions changes

        os.makedirs('./cache/results', exist_ok=True)
        with open(f"./cache/results/sensing_{n}.pkl", "wb") as f:
            pickle.dump(results, f)

    # # # interpret records
    # place = max(results, key=lambda x: results[x]['info_gain'])
    # print(f"Flood sensing at {place}")
    # marginals, joints = bayes_network_t.compress_multi_gmm({k: v['bn_updated'] for k, v in results.items()})

    # vis.map_roads_w_values(
    #     road_data.geo.copy(), results, city_shp_path=dir_city_boundary
    # )  # FIGURE 6: entropy map


"""
Validation
"""

for k in results.keys():
    speed_data = road_data.get_data_when_sensing_on(k)
    speed_calculator = mo.TrafficBayesNetwork(
        speed=road_data.speed_resampled, road_geo=road_data.geo, network_mode='causal',
        n_samples=10000, n_components=12, fitting_mode='one-off',
        remove_nodes=remove_data_from_nodes,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        speed_calculator.fit_marginal_from_data()

    _ = speed_calculator.calculate_network_kl_divergence([
        results[k]['bn_updated'][1]['marginals'],  # estimate
        speed_calculator.marginals,  # ground truth
    ])
    _ = speed_calculator.calculate_network_kl_divergence([
        bayes_network_t.marginals,  # historical
        speed_calculator.marginals,  # ground truth
    ])

    common_k = speed_calculator.marginals.keys() & results[k]['bn_updated'][1]['marginals'].keys()
    for kk in list(common_k):
        if np.array_equal(results[k]['bn_updated'][1]['marginals'][kk].means_,
                          bayes_network_t.marginals[kk].means_, ):
            continue
        vis.dist_gmm_1d(speed_calculator.marginals[kk])
        vis.dist_gmm_1d(results[k]['bn_updated'][1]['marginals'][kk])
        vis.dist_gmm_1d(bayes_network_t.marginals[kk])
        print(kk)

pass

