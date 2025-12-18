import data as dd
import model as mo
from config import *
import visualization as vis

import random
random.seed(0)
import numpy as np
np.random.seed(0)
df_random_seed = 0
import os
os.environ['PYTHONHASHSEED'] = str(0)
import pandas as pd


road_data = dd.RoadData()
road_data.load_instance(f'{dir_cache_instance}/road_data')
bayes_network_t = mo.TrafficBayesNetwork()
bayes_network_t.load_instance(f'{dir_cache_instance}/bayes_network_t')
bayes_network_t_normal = mo.TrafficBayesNetwork()
load_normal = bayes_network_t_normal.load_instance(f'{dir_cache_instance}/bayes_network_t_normal')

if not load_normal:
    # get traffic data during normal time
    normal_t_data = []
    for i in road_data.flood_time_citywide.copy().drop(
            road_data.flood_time_citywide.copy().sample(frac=0, random_state=df_random_seed).index
    ).index:
        rd_n = dd.RoadData()
        p = dd.get_period_before_start(road_data.flood_time_citywide.copy().iloc[[i]], 7)
        d_normal = rd_n.pull_nyc_dot_traffic_flooding(dir_NYC_data_token, select_incidents=p)
        if (d_normal is not None) and (len(d_normal) > 0):
            rd_n.resample_nyc_dot_traffic(rd_n.speed)
            normal_t_data.append(rd_n.speed_resampled)

    # fit priors from normal traffic data
    bayes_network_t_normal = mo.TrafficBayesNetwork(
        speed=pd.concat(normal_t_data), road_geo=road_data.geo, 
        n_samples=10000, n_components=12, corr_thr=corr_thr,
        remove_nodes=remove_data_from_nodes, 
    )
    # bayes_network_t_normal.speed_data = pd.concat(normal_t_data)
    bayes_network_t_normal.fit_marginal_from_data()
    bayes_network_t_normal.save_instance(f'{dir_cache_instance}/bayes_network_t_normal')

# validate on each incident
kld_d, kld_u, kld_i = [], [], []
all_inundations = []
all_covered = []
for i in road_data.flood_time_citywide.copy().index:

    # get inundated roads during the incident
    rd_test = dd.RoadData()  # empty instance
    inundation = rd_test.get_flooded_roads_during_inct(
        road_data.flood_time_per_road, # road time periods
        road_data.flood_time_citywide.copy().iloc[[i]]  # incident time period
    )
    if inundation is None or len(inundation) == 0:
        continue  # no inundation detected

    # filter inundations: bayes_network_t.marginals_downward[k] used
    if len([k for k in inundation if k in bayes_network_t.marginals_downward]) != len(inundation):
        continue  # flood time was determined using all data not filtered by corr, 150+ incidents were skipped

    # get traffic data during the incident
    d = rd_test.pull_nyc_dot_traffic_flooding(
        dir_NYC_data_token,
        select_incidents=road_data.flood_time_citywide.copy().iloc[[i]]
    )
    if d is None:
        continue  # the scope of modeling is based on aggregated data, data may be missing for some incidents
    rd_test.resample_nyc_dot_traffic(rd_test.speed)

    # fit marginals from the incident traffic data
    bn_t_test = mo.TrafficBayesNetwork(
        speed=rd_test.speed_resampled, road_geo=road_data.geo,
        n_samples=10000, n_components=12,
        remove_nodes=remove_data_from_nodes,
    )
    bn_t_test.fit_marginal_from_data()

    # estimate using the trained BN
    marginals_flood, update_loc_f, _, _ = bayes_network_t.update_network_with_multiple_soft_evidence(
        [{**{k: bn_t_test.marginals[k] for k in inundation}},  # signal_down
        {**{k: bn_t_test.marginals[k] for k in inundation}},],  # signal_up
        [bayes_network_t.marginals_downward.copy(), bayes_network_t.marginals_upward.copy()],  # marginals
        bayes_network_t.joints.copy(),  # joints
        verbose=0
    )
    all_inundations += inundation

    # eval
    print(f'Signal locs: {inundation}')
    # update_locs = update_loc_f['down'] + update_loc_f['up']  # include signal locs
    update_locs = [i for i in (update_loc_f['down'] + update_loc_f['up']) if i not in inundation]  # remove signal locs
    print(f'All traversed locs: {update_locs}')
    all_covered.append(update_locs)

    # compared disruption level among layouts
    estimated_disruption = bayes_network_t.calculate_network_kl_divergence([
        {k: v for k, v in marginals_flood.items() if k in update_locs},  # estimate
        {k: v for k, v in bayes_network_t_normal.marginals.items() if k in update_locs},  # averaged (normal period)
    ], expand=True 
    )
    kld_d.append([estimated_disruption, update_locs])

    # compared unexpectedness (without flood belief) among layouts
    estimated_unexpectedness = bayes_network_t.calculate_network_kl_divergence([
        {k: v for k, v in marginals_flood.items() if k in update_locs},  # estimate
        {k: v for k, v in bayes_network_t.marginals_downward.items() if k in update_locs},  # averaged (flood period)
    ], expand=True  
    )
    kld_u.append([estimated_unexpectedness, update_locs])


# # vis output


print('End of program.')
