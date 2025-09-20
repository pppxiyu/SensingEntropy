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


road_data = dd.RoadData()
load_r = road_data.load_instance('./cache/instances/road_data')
bayes_network_f = mo.FloodBayesNetwork()
load_f = bayes_network_f.load_instance('./cache/instances/bayes_network_f')
bayes_network_t = mo.TrafficBayesNetwork()
load_t = bayes_network_t.load_instance('./cache/instances/bayes_network_t')

# validate on each incident
kld = []
for i in road_data.flood_time_citywide.copy().drop(
        road_data.flood_time_citywide.copy().sample(frac=0, random_state=df_random_seed).index
).index:

    # get inundated roads during the incident
    rd_test = dd.RoadData()
    inundation = rd_test.get_flooded_roads_during_inct(
        road_data.flood_time_per_road, road_data.flood_time_citywide.copy().iloc[[i]]
    )
    # if len([r for r in inundation if r in bayes_network_t.signal_downward]) == 0: #!= len(inundation):
    #     continue  # all (or at least one) inundation should be able to enter BN

    # get traffic data in the incident
    d = rd_test.pull_nyc_dot_traffic_flooding(
        dir_NYC_data_token,
        select_incidents=road_data.flood_time_citywide.copy().iloc[[i]]
    )
    if d is None:
        continue
    rd_test.resample_nyc_dot_traffic(rd_test.speed)

    # fit signals from the incident data
    bn_t_test = mo.TrafficBayesNetwork(
        speed=rd_test.speed_resampled, road_geo=road_data.geo, network_mode='causal',
        n_samples=10000, n_components=12, fitting_mode='one-off',
        remove_nodes=remove_data_from_nodes,
    )
    bn_t_test.fit_joint()
    bn_t_test.fit_marginal_from_joints(upward=False)
    bn_t_test.fit_marginal_from_joints(upward=True)
    bn_t_test.fit_signal(
        road_data.flood_time_per_road, bn_t_test.marginals_downwards,
        mode='from_marginal', upward=False, signal_filter=.025
    )

    # using the real world signals to update BN
    signals = inundation.copy()
    inferred_inundation = [
        bn_t_test.convert_state_to_dist(
            bayes_network_f.infer_node_states(s, 1, 1, 1)
        ) for s in signals
    ]

    # if len([k for k in signals if k in bn_t_test.marginals_downward]) != len(signals):
    #     continue  # all required signals should have the real world observation

    marginals_flood, update_loc_f = bayes_network_t.update_network_with_multiple_soft_evidence(
        [
            {**{k: bn_t_test.marginals_downward[k] for k in signals},
             **{k: v for i in inferred_inundation if i[0] != {} for k, v in i[0].items()}},  # signal_down
            {**{k: bn_t_test.marginals_upward[k] for k in signals},
             **{k: v for i in inferred_inundation if i[1] != {} for k, v in i[1].items()}},  # signal_up
        ],
        [
            bayes_network_t.marginals_downward.copy(),
            bayes_network_t.marginals_upward.copy()
        ],  # marginals
        bayes_network_t.joints.copy(),  # joints
        verbose=0
    )

    # eval
    print(f'Signal locs: {signals}')
    update_locs = update_loc_f['down'] + update_loc_f['up']
    print(f'All traversed locs: {update_locs}')
    estimated = bayes_network_t.calculate_network_kl_divergence([
        {k: v for k, v in marginals_flood.items() if k in update_locs},  # estimate
        {k: v for k, v in bn_t_test.marginals_downward.items() if k in update_locs},  # ground truth
    ])
    historical = bayes_network_t.calculate_network_kl_divergence([
        {k: v for k, v in bayes_network_t.marginals_downward.items() if k in update_locs},  # averaged
        {k: v for k, v in bn_t_test.marginals_downward.items() if k in update_locs},  # ground truth
    ])
    kld.append([historical, estimated])

kld = [i for i in kld if i[0] is not None]
relative_changes = [(e1 - e2) / e1 if e1 != 0 else 0 for e1, e2 in kld]

print('End of program.')

