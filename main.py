from config import *
import data as dd
import model as mo
import visualization as vis

import random
import numpy as np

import pickle
import os
import warnings

random.seed(0)
np.random.seed(0)
df_random_seed = 0
os.environ['PYTHONHASHSEED'] = str(0)


"""
Load cache
"""
road_data = dd.RoadData()
load_r = road_data.load_instance(f'{dir_cache_instance}/road_data')
bayes_network_f = mo.FloodBayesNetwork()
load_f = bayes_network_f.load_instance(f'{dir_cache_instance}/bayes_network_f')
bayes_network_t = mo.TrafficBayesNetwork()
load_t = bayes_network_t.load_instance(f'{dir_cache_instance}/bayes_network_t')
bayes_network_t_normal = mo.TrafficBayesNetwork()
load_normal = bayes_network_t_normal.load_instance(f'{dir_cache_instance}/bayes_network_t_normal')
assert load_normal == True

if not load_r:
    """
    Data
    """
    road_data = dd.RoadData()
    # Geometry manual processing
    # road_data.pull_nyc_dot_traffic(
    #     dir_NYC_data_token, ['2020-04-01', '2020-04-30'], True
    # )  # road geometries have been downloaded, checked, and corrected. Comment this line.
    road_data.import_adapted_nyc_road(dir_adapted_nyc_roads)

    # Identify flooding periods from df
    road_data.import_street_flooding(dir_road_closure, local_crs, 20)
    # vis.map_roads_n_flood_plt(
    #     road_data.geo.to_crs(local_crs), road_data.df.to_crs(local_crs), buffer=20,
    # )  # FIGURE X: flood point and roads
    # vis.map_flood_p(road_data.geo, road_data.closure_p_per_segment, mapbox_token, local_crs)  # FIGURE X: flood p
    road_data.infer_flooding_time_citywide(pre_flood_buffer=4, post_flood_buffer=8)
    road_data.infer_flooding_time_per_road(pre_flood_buffer=4, post_flood_buffer=8)

    # Pull traffic data in flooding periods
    road_data.pull_nyc_dot_traffic_flooding(
        dir_NYC_data_token,
        select_incidents=road_data.flood_time_citywide.copy()
    )
    road_data.resample_nyc_dot_traffic(road_data.speed)

    road_data.save_instance(f'{dir_cache_instance}/road_data')

if not load_f:
    """
    Modeling
    """
    # Fit marginals - flood
    bayes_network_f = mo.FloodBayesNetwork()
    bayes_network_f.fit_marginal(road_data.closures)
    # vis.bar_flood_prob(bayes_network_f.marginals.iloc[48])  # FIGURE 5: flood probability

    # Fit conditionals - flood
    bayes_network_f.build_network_by_co_occurrence(road_data.closures, weight_thr=0.2)
    bayes_network_f.fit_conditional(road_data.closures)
    bayes_network_f.build_bayes_network()

    bayes_network_f.save_instance(f'{dir_cache_instance}/bayes_network_f')

if not load_t:
    # Init
    bayes_network_t = mo.TrafficBayesNetwork(
        speed=road_data.speed_resampled, road_geo=road_data.geo, network_mode='causal',
        n_samples=10000, n_components=12, fitting_mode='one-off',
        remove_nodes=remove_data_from_nodes, corr_thr=corr_thr, 
    )
    # vis.map_roads_n_topology_plt(
    #     geo_roads=road_data.geo.copy(),
    #     network=bayes_network_t.network.copy(), network_geo_roads=road_data.geo.copy(),
    #     local_crs=local_crs, city_shp_path=dir_city_boundary,
    # )  # FIGURE 1: road network and topology

    # Fit joints - speed
    bayes_network_t.fit_joint()
    # for k, v in list(bayes_network_t.joints.items()):
    #     if len(v['parents']) == 1:  # the vis only support 3d
    #         print(f'Joint distributions at {k}')
    #         vis.dist_gmm_3d(v['joints'], k, v['parents'][0])  # FIGURE 2: joint distributions between roads

    # Fit marginals - speed
    bayes_network_t.fit_marginal_from_data()
    bayes_network_t.fit_marginal_from_joints(upward=False)
    bayes_network_t.fit_marginal_from_joints(upward=True)
    # for seg in road_data.speed['link_id'].unique()[0: 10]:
    #     vis.dist_histo_gmm_1d(
    #         road_data.speed[road_data.speed['link_id'] == seg]['speed'].values,
    #         bayes_network_t.marginals[seg]
    #     )  # FIGURE X: marginal distributions on roads

    # Fit signals
    bayes_network_t.fit_signal(
        road_data.flood_time_per_road, bayes_network_f.marginals,
        mode='from_marginal', upward=False, 
    )
    bayes_network_t.fit_signal(
        road_data.flood_time_per_road, bayes_network_f.marginals,
        mode='from_marginal', upward=True,
    )
    # for k, v in bayes_network_t.signal_downward.items():
    #     print(f'Distributions under observations at {k}')
    #     vis.dist_discrete_gmm(v, bayes_network_t.marginals[k],)  # FIGURE 3: distribution with observation

    bayes_network_t.check_keys()
    bayes_network_t.save_instance(f'{dir_cache_instance}/bayes_network_t')
# mo.check_gmr_bn_consistency(list(bayes_network_t.marginals.keys()), bayes_network_t.joints)

"""
Placement
"""
# Update network with observation
for n in range(sensor_count):

    # get records
    try:
        with open(f"{dir_results}/sensing_{n}.pkl", "rb") as f:
            results = pickle.load(f)
            print(f"Loaded sensing_{n} from disk.")
    except (FileNotFoundError, pickle.UnpicklingError, EOFError) as e:
        print(f"Loading failed ({e}), continuing with calculation.")

        results = {}
        for k, v in bayes_network_t.signal_downward.items():
            inferred_signals_flood = bayes_network_t.convert_state_to_dist(
                bayes_network_f.infer_node_states(k, 1, 1, 1)
            )
            signals = [
                    {**{k: v['speed_flood']}, **inferred_signals_flood[0]},  # downward
                    {**{k: bayes_network_t.signal_upward[k]['speed_flood']}, **inferred_signals_flood[1]},  # upward
                ]
            marginals_flood, update_loc_f = bayes_network_t.update_network_with_multiple_soft_evidence(
                signals,
                [bayes_network_t.marginals_downward.copy(), bayes_network_t.marginals_upward.copy()],
                bayes_network_t.joints.copy(),
                verbose=0
            )
            
            bn_updated = {'p': v['p_flood'], 'marginals': marginals_flood}
            covered_locs = [*update_loc_f['down'], *update_loc_f['up']]
            prior = {k: v for k, v in bayes_network_t.marginals_downward.copy().items() if k in covered_locs}
            normal = {k: v for k, v in bayes_network_t_normal.marginals.copy().items() if k in covered_locs}
            
            results[k] = {
                'road': k,
                'flood_p': bn_updated['p'],
                'updated_marginals': bn_updated['marginals'],
                'covered_locs': covered_locs,
                'signal_strength':[
                    mo.calculate_kl_divergence_gmm(v, bayes_network_t_normal.marginals[k]) 
                    if k in bayes_network_t_normal.marginals.keys() else None for s in signals for k, v in s.items()
                    ],
                'info_gain_suprise': bayes_network_t.calculate_network_kl_divergence(
                    [{k: v for k, v in marginals_flood.items() if k in covered_locs}, prior], label=k
                ),
                'info_gain_disruption': bayes_network_t.calculate_network_kl_divergence(
                    [{k: v for k, v in marginals_flood.items() if k in covered_locs},  normal], label=k
                ),
            }
            results[k]['info_gain_disruption_weighted'] = (results[k]['flood_p'] * results[k]['info_gain_disruption'] 
                                                           if results[k]['info_gain_disruption'] is not None else None)
            
            # if (k in joints_flood.keys()) and (len(joints_flood[k][0]) == 1):
            #     z_max = vis.dist_gmm_3d(joints_flood[k][1], k, joints_flood[k][0][0], return_z_max=True)
            #     vis.dist_gmm_3d(
            #         bayes_network_t.joints[k][1], k, bayes_network_t.joints[k][0][0],
            #         z_limit=z_max
            #     )  # FIGURE 4: joint distributions changes

        os.makedirs(f'{dir_results}', exist_ok=True)
        with open(f"{dir_results}/sensing_{n}.pkl", "wb") as f:
            pickle.dump(results, f)

        selected_roads, (_, v_disruption, v_suprise), processed_results = mo.process_results(results, weight_disruption)
        
        # the higher the flood p is, the higher the disruption captured is
        vis.scatter_disp_n_supr_w_flood_p([v[2]['flood_p'] for v in processed_results], v_disruption)
        # the higher the flood p is, the lower the suprise captured is
        vis.scatter_disp_n_supr_w_flood_p([v[2]['flood_p'] for v in processed_results], v_suprise)
        # the longer the propagation is, the higher the disruption captured is
        vis.scatter_disp_n_supr_w_flood_p([len(v[2]['covered_locs']) for v in processed_results], v_disruption)
        # the longer the propagation is, the higher the suprise captured is
        vis.scatter_disp_n_supr_w_flood_p([len(v[2]['covered_locs']) for v in processed_results], v_suprise)
        # the stronger the signal is, the higher the disruption captured is
        vis.scatter_disp_n_supr_w_flood_p([np.mean(v[2]['signal_strength']) for v in processed_results], v_disruption)
        # the stronger the signal is, the higher the suprise captured is
        vis.scatter_disp_n_supr_w_flood_p([np.mean(v[2]['signal_strength']) for v in processed_results], v_suprise)
        
        # vis.map_roads_w_values(
        #     road_data.geo.copy(), results, city_shp_path=dir_city_boundary
        # )  # FIGURE 6: entropy map



print('End of program.')
