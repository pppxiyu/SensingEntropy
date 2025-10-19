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
assert load_normal == True, 'Save the normal time BN using val.py first'

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
    road_data.infer_flooding_time_citywide(pre_flood_buffer=2, post_flood_buffer=8)
    road_data.infer_flooding_time_per_road(pre_flood_buffer=2, post_flood_buffer=8)

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

    # Fit joints - speed
    bayes_network_t.fit_joint()

    # Fit marginals - speed
    bayes_network_t.fit_marginal_from_data()
    bayes_network_t.fit_marginal_from_joints(upward=False)
    bayes_network_t.fit_marginal_from_joints(upward=True)

    # Fit signals
    bayes_network_t.fit_signal(
        road_data.flood_time_per_road, bayes_network_f.marginals,
        mode='from_marginal', upward=False, 
    )
    bayes_network_t.fit_signal(
        road_data.flood_time_per_road, bayes_network_f.marginals,
        mode='from_marginal', upward=True,
    )

    bayes_network_t.check_keys()
    bayes_network_t.save_instance(f'{dir_cache_instance}/bayes_network_t')
# mo.check_gmr_bn_consistency(list(bayes_network_t.marginals.keys()), bayes_network_t.joints)


# # visualizations
# vis.map_roads_n_flood(
#     road_data.geo.to_crs(local_crs), road_data.closures.to_crs(local_crs), buffer=20,
# )  # flood point and roads

# vis.bar_flood_prob(bayes_network_f.marginals.iloc[48])  # flood probability of a road

# vis.map_roads_n_values(
#     road_data.geo, bayes_network_f.marginals.set_index('link_id')['p'].to_dict(), 
#     local_crs, city_shp_path=dir_city_boundary, y_label='Flood probability', 
#     save_dir=f'{dir_figures}/map_flood_p.png',
# )  # flood probability map

# vis.map_roads_n_topology(
#     geo_roads=road_data.geo.copy(),
#     network=bayes_network_f.network.copy(), network_geo_roads=road_data.geo.copy(),
#     local_crs=local_crs, city_shp_path=dir_city_boundary, shift=False, save_dir=f'{dir_figures}/map_flood_bn_topolgy.png',
# )  # flood bayes network edges

# vis.map_roads_n_topology(
#     geo_roads=road_data.geo.copy(),
#     network=bayes_network_t.network.copy(), network_geo_roads=road_data.geo.copy(),
#     local_crs=local_crs, city_shp_path=dir_city_boundary, save_dir=f'{dir_figures}/map_traffic_bn_topolgy.png',
# )  # traffic bayes network edges

# for k, v in list(bayes_network_t.joints.items()):
#     if k == '4362252':
#         if len(v['parents']) == 1:  # the vis only support 3d
#             print(f'Joint distributions at {k}')
#             vis.dist_gmm_3d(v['joints'], k, v['parents'][0], save_dir=f'{dir_figures}/dist_joint_4362252')  # joint distributions between roads

# for k, v in bayes_network_t.marginals_downward.items():
#     if k in ['4362252', '4456483']:
#         vis.dist_gmm_1d(v, save_dir=f'{dir_figures}/dist_marginal_{k}')  # marginal distributions on roads

# for k, v in bayes_network_t.signal_downward.items():
#     print(f'Distributions under observations at {k}')
#     vis.dist_discrete_gmm(v, bayes_network_t.marginals[k],)  # distribution with observation

"""
Placement for Sensor 1
"""
# Update network with observation for the 1st sensor
# get records
try:
    with open(f"{dir_results}/placement/sensing_1st_round.pkl", "rb") as f:
        results_0 = pickle.load(f)
        print(f"Loaded sensing_1st_round from disk.")
    with open(f"{dir_results}/placement/sensing_1.pkl", "rb") as f:
        sensor_1 = pickle.load(f)
        print(f"Loaded sensing_1 from disk.")
except (FileNotFoundError, pickle.UnpicklingError, EOFError) as e:
    print(f"Loading failed ({e}), continuing with calculation.")

    print(f'\nCalculation for the Sensor 1')
    results_0 = {}
    for k, v in bayes_network_t.signal_downward.items():
        # signals
        signals = mo.get_signals(k, v, bayes_network_t, bayes_network_f)
        
        # update
        marginals_combined, update_loc_f, joints_flood, marginals_flood = bayes_network_t.update_network_with_multiple_soft_evidence(
            signals,
            [bayes_network_t.marginals_downward.copy(), bayes_network_t.marginals_upward.copy()],
            bayes_network_t.joints.copy(),
            verbose=0
        )
        
        # record and analyze results
        covered_locs_k = [*update_loc_f['down'], *update_loc_f['up']]
        prior = {k: v for k, v in bayes_network_t.marginals_downward.copy().items() if k in covered_locs_k}
        normal = {k: v for k, v in bayes_network_t_normal.marginals.copy().items() if k in covered_locs_k}
        results_0[k] = {
            'road': k,
            'flood_p': v['p_flood'],
            'signals': signals,
            'updated_marginal_w_direction': marginals_flood,
            'updated_joints': joints_flood,
            'covered_locs_raw': update_loc_f,
            'covered_locs': covered_locs_k,
            'signal_strength':[
                mo.calculate_kl_divergence_gmm(v, bayes_network_t_normal.marginals[k]) 
                if k in bayes_network_t_normal.marginals.keys() else None for s in signals for k, v in s.items()
                ],
            'info_gain_suprise': bayes_network_t.calculate_network_kl_divergence(
                [{k: v for k, v in marginals_combined.items() if k in covered_locs_k}, prior], label=k
            ),
            'info_gain_disruption': mo.safe_subtract(bayes_network_t.calculate_network_kl_divergence(
                [{k: v for k, v in marginals_combined.items() if k in covered_locs_k},  normal], label=k
            ), (bayes_network_t.calculate_network_kl_divergence(
                [{k: v for k, v in prior.items() if k in covered_locs_k},  normal], label=k
            ))),
        }
        results_0[k]['info_gain_disruption_weighted'] = (
            results_0[k]['flood_p'] * results_0[k]['info_gain_disruption'] 
            if results_0[k]['info_gain_disruption'] is not None else None
        )

    selected_road, results_0 = mo.norm_n_weight(results_0, weight_disruption)
    sensor_1 = results_0[selected_road].copy()

    os.makedirs(f'{dir_results}/placement', exist_ok=True)
    with open(f"{dir_results}/placement/sensing_1st_round.pkl", "wb") as f:
        pickle.dump(results_0, f)
    with open(f"{dir_results}/placement/sensing_1.pkl", "wb") as f:
        pickle.dump(sensor_1, f)

"""
Visualizations
"""
# # joints change
# for k in results_0.keys():
#     if k not in ['4616272', '4616329', '4616351']:
#         continue
#     closest_loc_down, closest_loc_up = None, None
#     if results_0[k]['covered_locs_raw']['down']:
#         closest_loc_down_list = [i for i in results_0[k]['covered_locs_raw']['down'] if i != k]
#         if closest_loc_down_list != []:
#             closest_loc_down = closest_loc_down_list[0]
#     if results_0[k]['covered_locs_raw']['up']:
#         closest_loc_up = results_0[k]['covered_locs_raw']['up'][0]
    
#     if closest_loc_down is not None:
#         vis.dist_discrete_gmm(
#             bayes_network_t.signal_downward[k], save_dir=f'{dir_figures}/working_example_{k}/down_signal_both.png'
#         ) # show both positive and negative signal
#         vis.dist_gmm_1d(
#             results_0[k]['signals'][0][k], save_dir=f'{dir_figures}/working_example_{k}/down_flood_signal.png'
#         )  # signal resulting in the change
#         joints_closest_loc_down = results_0[k]['updated_joints'][closest_loc_down]['joints']
#         z_max = vis.dist_gmm_3d(
#             joints_closest_loc_down, closest_loc_down, k, return_z_max=True,
#             save_dir=f'{dir_figures}/working_example_{k}/down_updated_joints.png', switch_axis=True
#         )  # updated joints
#         vis.dist_gmm_3d(
#             bayes_network_t.joints[closest_loc_down]['joints'], closest_loc_down, k,
#             z_limit=z_max, save_dir=f'{dir_figures}/working_example_{k}/down_original_joints.png', switch_axis=True
#         )  # original joints
#         y_max = vis.dist_gmm_1d(
#             results_0[k]['updated_marginal_w_direction'][0][closest_loc_down], 
#             save_dir=f'{dir_figures}/working_example_{k}/down_updated_marginal_{closest_loc_down}.png',
#             return_y_max=True,
#         )  # updated marginal of the next node
#         vis.dist_gmm_1d(
#             bayes_network_t.marginals_downward[closest_loc_down], 
#             save_dir=f'{dir_figures}/working_example_{k}/down_original_marginal_{closest_loc_down}.png',
#             y_limit=y_max,
#         )  # original marginal of the next node

#     if closest_loc_up is not None:
#         vis.dist_discrete_gmm(
#             bayes_network_t.signal_upward[k], save_dir=f'{dir_figures}/working_example_{k}/up_signal_both.png'
#         )  # show both positive and negative signal
#         vis.dist_gmm_1d(
#             results_0[k]['signals'][1], save_dir=f'{dir_figures}/working_example_{k}/up_flood_signal.png', switch_axis=True
#         )  # signal resulting in the change        
#         joints_closest_loc_up = results_0[k]['updated_joints'][k]['joints']
#         z_max = vis.dist_gmm_3d(
#             joints_closest_loc_up, k, closest_loc_up, return_z_max=True,
#             save_dir=f'{dir_figures}/working_example_{k}/up_updated_joints.png'
#         )  # updated joints
#         vis.dist_gmm_3d(
#             bayes_network_t.joints[k]['joints'], k, closest_loc_up,
#             z_limit=z_max, save_dir=f'{dir_figures}/working_example_{k}/up_original_joints.png', switch_axis=True
#         )  # original joints
#         y_max = vis.dist_gmm_1d(
#             results_0[k]['updated_marginal_w_direction'][1][closest_loc_up], 
#             save_dir=f'{dir_figures}/working_example_{k}/up_updated_marginal_{closest_loc_up}.png',
#             return_y_max=True,
#         )  # updated marginal of the next node
#         vis.dist_gmm_1d(
#             bayes_network_t.marginals_upward[closest_loc_up], 
#             save_dir=f'{dir_figures}/working_example_{k}/up_original_marginal_{closest_loc_up}.png',
#             y_limit=y_max,
#         )  # original marginal of the next node

# # VoI Map
# vis.map_roads_n_values(
#     road_data.geo.copy()[road_data.geo.copy()['link_id'].isin(bayes_network_t.network.nodes())], 
#     {k: v['voi'] for k, v in results_0.items()}, city_shp_path=dir_city_boundary,
#     save_dir=f'{dir_figures}/map_voi.png', enlarge_font=True,
#     y_label='Weighted $VoI$', coord_decimals=2, cmap='cosmic', 
# )  # VoI map
# vis.map_roads_n_values(
#     road_data.geo.copy()[road_data.geo.copy()['link_id'].isin(bayes_network_t.network.nodes())], 
#     {k: v['info_gain_disruption_weighted_normed'] for k, v in results_0.items()}, city_shp_path=dir_city_boundary,
#     save_dir=f'{dir_figures}/map_d.png', enlarge_font=True,
#     y_label='Disruption ($D$)', coord_decimals=2, cmap='emerald'
# )  # VoI map
# vis.map_roads_n_values(
#     road_data.geo.copy()[road_data.geo.copy()['link_id'].isin(bayes_network_t.network.nodes())], 
#     {k: v['info_gain_suprise_normed'] for k, v in results_0.items()}, city_shp_path=dir_city_boundary,
#     save_dir=f'{dir_figures}/map_uod.png', enlarge_font=True,
#     y_label='Unexpectedness ($UoD$)', coord_decimals=2, cmap='sapphire', 
# )  # VoI map

# # Placement insights
# # the higher the flood p is, the higher the disruption captured is
# vis.scatter_disp_n_supr_w_factors(
#     [v['flood_p'] for _, v in results_0.items()], [v['info_gain_disruption_weighted_normed'] for _, v in results_0.items()],
#     xlabel="Flood probability\n", ylabel="Normalized disruption\n($D$) captured",
#     save_dir=f'{dir_figures}/scatter_flood_p_n_disruption.png',
#     dot_color='#6F8ABF', line_color='#243E73', 
# )
# # the higher the flood p is, the lower the suprise captured is
# vis.scatter_disp_n_supr_w_factors(
#     [v['flood_p'] for _, v in results_0.items()], [v['info_gain_suprise_normed'] for _, v in results_0.items()],
#     xlabel="Flood probability\n", ylabel="Normalized unexpectedness\n($UoD$) captured",
#     save_dir=f'{dir_figures}/scatter_flood_p_n_unexpectedness.png',
#     dot_color='#8B6FBF', line_color='#4B3E73', 
# )
# # the longer the propagation is, the higher the disruption captured is
# vis.scatter_disp_n_supr_w_factors(
#     [len(v['covered_locs']) for _, v in results_0.items()], [v['info_gain_disruption_weighted_normed'] for _, v in results_0.items()],
#     xlabel="Propagation distance\n(# nodes)", ylabel="Normalized disruption\n($D$) captured",
#     save_dir=f'{dir_figures}/scatter_propagation_dist_n_disruption.png',
#     dot_color='#6F8ABF', line_color='#243E73', legend_loc='upper right', 
# )
# # the longer the propagation is, the higher the suprise captured is
# vis.scatter_disp_n_supr_w_factors(
#     [len(v['covered_locs']) for _, v in results_0.items()], [v['info_gain_suprise_normed'] for _, v in results_0.items()],
#     xlabel="Propagation distance\n(# nodes)", ylabel="Normalized unexpectedness\n($UoD$) captured",
#     save_dir=f'{dir_figures}/scatter_propagation_dist_n_unexpectedness.png',
#     dot_color='#8B6FBF', line_color='#4B3E73', legend_loc='upper right', 
# )
# # the stronger the signal is, the higher the disruption captured is
# vis.scatter_disp_n_supr_w_factors(
#     [np.mean(v['signal_strength']) for _, v in results_0.items()], [v['info_gain_disruption_weighted_normed'] for _, v in results_0.items()],
#     xlabel="Signal strength\n( $KLD$(prior, signal) )", ylabel="Normalized disruption\n($D$) captured",
#     save_dir=f'{dir_figures}/scatter_signal_strength_n_disruption.png',
#     dot_color='#6F8ABF', line_color='#243E73', 
# )
# # the stronger the signal is, the higher the suprise captured is
# vis.scatter_disp_n_supr_w_factors(
#     [np.mean(v['signal_strength']) for _, v in results_0.items()], [v['info_gain_suprise_normed'] for _, v in results_0.items()],
#     xlabel="Signal strength\n( $KLD$(prior, signal) )", ylabel="Normalized unexpectedness\n($UoD$) captured",
#     save_dir=f'{dir_figures}/scatter_signal_strength_n_unexpectedness.png',
#     dot_color='#8B6FBF', line_color='#4B3E73', 
# )


"""
Placement for multiple sensors
"""
try:  # try to access info for all other sensors
    with open(f"{dir_results}/placement/sensors_2nd_n_all.pkl", "rb") as f:
        sensors_2nd_n_all = pickle.load(f)
        print(f"Loaded sensor info except for the 1st one from disk.")
except (FileNotFoundError, pickle.UnpicklingError, EOFError) as e:
    print(f"Loading failed ({e}), continuing with calculation.")
    sensors_2nd_n_all = []
    
    # init
    thr_p = min([i['p_flood'] for i in list(bayes_network_t.signal_downward.values())]) * prune_rate
    norm_d_lower_bound = min({k: v['info_gain_disruption_weighted'] for k, v in results_0.items()}.values())
    norm_d_upper_bound = max({k: v['info_gain_disruption_weighted'] for k, v in results_0.items()}.values())
    norm_uod_lower_bound = min({k: v['info_gain_suprise'] for k, v in results_0.items()}.values())
    norm_uod_upper_bound = max({k: v['info_gain_suprise'] for k, v in results_0.items()}.values())

    # iter
    belief_networks = [
        (sensor_1['updated_joints'], sensor_1['updated_marginal_w_direction'], sensor_1['flood_p']), 
        (bayes_network_t.joints.copy(), [bayes_network_t.marginals_downward.copy(), bayes_network_t.marginals_upward.copy()], 1 - sensor_1['flood_p'])
    ]
    placed = [sensor_1['road']]

    for n in range(sensor_count - 1):
        print(f'\nCalculation for the Sensor {n + 2}')
        results = {}  # temporary dict for recording
        # iterate all unused locs for placing the next sensor
        for k, v in bayes_network_t.signal_downward.items():
            if k in placed:
                continue

            new_belief_networks = []
            marginals_only = []  # all bn updated by the flooding signal
            signals = mo.get_signals(k, v, bayes_network_t, bayes_network_f)
            for (joints_current, marginals_current, p_current) in belief_networks:
                
                # Calculate branch probability
                p_branch = p_current * v['p_flood']
                if p_branch >= thr_p:
                    # get updated BN when sensor is on and add to new set of belief networks
                    marginals_combined, update_loc_f, joints_updated, marginals_updated = bayes_network_t.update_network_with_multiple_soft_evidence(
                        signals, marginals_current, joints_current, verbose=0
                    )
                    new_belief_networks.append((joints_updated, marginals_updated, p_branch))
                    marginals_only.append((marginals_combined, update_loc_f, p_branch))

                # Calculate branch probability
                p_branch = p_current * (1 - v['p_flood'])
                if p_branch >= thr_p:
                    # get updated BN when sensor is off (keep original) and add to new set of belief networks
                    new_belief_networks.append((joints_current, marginals_current, p_branch))

            # check and edit belief_network_4_compute
            multi_marginals_only = mo.edit_marginals_only(marginals_only,)

            # baseline remains the same
            prior = {
                k: v for k, v in bayes_network_t.marginals_downward.copy().items() if k in marginals_only[0][1]['down']
            }
            prior.update({
                k: v for k, v in bayes_network_t.marginals_upward.copy().items() if k in marginals_only[0][1]['up']
            })
            normal = {
                k: v for k, v in bayes_network_t_normal.marginals.copy().items() if k in 
                (marginals_only[0][1]['down'] + marginals_only[0][1]['up'])
            }

            # combine the scenarios that this sensor is conditioned on
            multi_belief_networks_marginals = mo.edit_belief_networks(
                belief_networks.copy(), marginals_only[0][1]['down'], marginals_only[0][1]['up']
            )

            # record
            results[k] = {
                'road': k,
                'flood_p': v['p_flood'],
                'belief_networks': new_belief_networks,
                'covered_locs_k': marginals_only[0][1]['down'] + marginals_only[0][1]['up'], #covered_locs_k,
                # 'info_gain_suprise': mo.safe_subtract(bayes_network_t.calculate_network_kl_divergence(
                #     [multi_marginals_only, prior], label=k
                # ), bayes_network_t.calculate_network_kl_divergence(
                #     [multi_belief_networks_marginals, prior], label=k, verbose=0
                # )),  # this calculation would also work
                'info_gain_suprise':bayes_network_t.calculate_network_kl_divergence(
                    [multi_marginals_only, multi_belief_networks_marginals], label=k
                ),  # this calculation is more correct than the one above, as KLD is not strictly additive
                'info_gain_disruption': mo.safe_subtract(bayes_network_t.calculate_network_kl_divergence(
                    [multi_marginals_only, normal], label=k
                ), bayes_network_t.calculate_network_kl_divergence(
                    [multi_belief_networks_marginals, normal], label=k, verbose=0
                )),
            }
            results[k]['info_gain_disruption_weighted'] = (
                results[k]['flood_p'] * results[k]['info_gain_disruption'] 
                if results[k]['info_gain_disruption'] is not None else None
            )

        # select from the temporary records
        selected_road, results = mo.norm_n_weight(
            results, weight_disruption, 
            [norm_d_lower_bound, norm_d_upper_bound, norm_uod_lower_bound, norm_uod_upper_bound]
        )

        # add the selection to the results_2nd_n_all
        sensors_2nd_n_all.append(results[selected_road])

        # update iterates
        belief_networks = results[selected_road]['belief_networks']
        placed.append(selected_road)

    # save results except for the 1st one
    os.makedirs(f'{dir_results}/placement', exist_ok=True)
    with open(f"{dir_results}/placement/sensors_2nd_n_all.pkl", "wb") as f:
        pickle.dump(sensors_2nd_n_all, f)

"""
Visualizations
"""
# # metrics trends
# vis.line_placement_d_n_uod(
#     [sensor_1['info_gain_disruption_weighted']] + [i['info_gain_disruption_weighted'] for i in sensors_2nd_n_all],
#     np.cumsum([sensor_1['info_gain_disruption_weighted']] + [i['info_gain_disruption_weighted'] for i in sensors_2nd_n_all]).tolist(),
#     xlabel=r"The $n^{th}$ sensing", ylabel_left="Disruption ($D$)\ncaptured", ylabel_right="Accumulated\ndisruption ($D$)\n",
#     save_dir=f'{dir_figures}/line_sensing_vs_d_accum.png',
# )
# vis.line_placement_d_n_uod(
#     [sensor_1['info_gain_suprise']] + [i['info_gain_suprise'] for i in sensors_2nd_n_all],
#     np.cumsum([sensor_1['info_gain_suprise']] + [i['info_gain_suprise'] for i in sensors_2nd_n_all]).tolist(),
#     xlabel=r"The $n^{th}$ sensing", ylabel_left="Unexpectedness\n($UoD$) captured", ylabel_right="Accumulated\n unexpectedness\n($UoD$)",
#     save_dir=f'{dir_figures}/line_sensing_vs_uod_accum.png',
# )
# vis.line_placement_d_n_uod(
#     [sensor_1['voi']] + [i['voi'] for i in sensors_2nd_n_all],
#     np.cumsum([sensor_1['voi']] + [i['voi'] for i in sensors_2nd_n_all]).tolist(),
#     xlabel=r"The $n^{th}$ sensing", ylabel_left="\n$VoI$ captured", ylabel_right="Accumulated $VoI$\n\n",
#     save_dir=f'{dir_figures}/line_sensing_vs_voi_accum.png',
# )

# # compare with other strategies
# """
# Note that the alternative strategy implementations here are oversimplified. The results will be the same
# in our datasets, because there is sensing-wise interaction in the specific dataset. The correct way to implement
# is specifying the sensing one by one, and updating the network after each sensing, similar to the main strategy.
# """
# all_sensor = [sensor_1] + sensors_2nd_n_all
# rank_by_flood_p = sorted(all_sensor, key=lambda x: x['flood_p'], reverse=True)
# rank_by_traffic = [
#     {'road': i, 'speed': road_data.speed_resampled[road_data.speed_resampled['link_id'] == i]['speed'].mean(), 'voi':j} 
#     for i, j in [(i['road'], i['voi']) for i in all_sensor]
# ]
# rank_by_traffic = sorted(rank_by_traffic, key=lambda x: x['speed'], reverse=False)
# vis.line_multi_strategy([
#         np.cumsum([sensor_1['voi']] + [i['voi'] for i in sensors_2nd_n_all]).tolist(),
#         np.cumsum([i['voi'] for i in rank_by_flood_p]).tolist(),
#         np.cumsum([i['voi'] for i in rank_by_traffic]).tolist(),
#     ], ['Priorting $VoI$', 'Priorting flood probability', 'Priorting traffic volume'],
#     xlabel=r"The $n^{th}$ sensing", ylabel_left="\n$VoI$ captured",
#     save_dir=f'{dir_figures}/line_multi-strategy.png', y_buffer=0.1
# )

print('End of program.')
