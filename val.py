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

kld = []
rd_test = dd.RoadData()
for i in road_data.flood_time_citywide.copy().drop(
        road_data.flood_time_citywide.copy().sample(frac=0, random_state=df_random_seed).index
).index:

    # check if signals for the inundation are available
    inundation = rd_test.get_flooded_roads_during_inct(
        road_data.flood_time_per_road, road_data.flood_time_citywide.copy().iloc[[i]]
    )
    available = False
    for r in inundation:
        if r in bayes_network_t.signal_downward:
            available = True
            break
    if not available:
        continue

    """
    temporary codes (for locate the bad inference)
    Signals 4456494 abd 4616272 appears 4 time separately and fails 1 time out of the 4. The reason of failure 
    is that the signal from historical average is too different from the distribution of the specific incident.
    Signal 4620343 appears once and failed. The reason is that the dependency based on all data does not 
    work for the specific incidents.
    """
    # signals = [r for r in inundation if r in bayes_network_t.signal_downward]
    # if not any(elem in signals for elem in ['4620343']):
    #     continue

    # get incident data and ground truth
    d = rd_test.pull_nyc_dot_traffic_flooding(
        dir_NYC_data_token,
        select_incidents=road_data.flood_time_citywide.copy().iloc[[i]]
    )
    if d is None:
        continue
    rd_test.resample_nyc_dot_traffic(rd_test.speed)
    bn_t_test = mo.TrafficBayesNetwork(
        speed=rd_test.speed_resampled, road_geo=road_data.geo, network_mode='causal',
        n_samples=10000, n_components=12, fitting_mode='one-off',
        remove_nodes=remove_data_from_nodes,
    )
    bn_t_test.fit_joint()
    bn_t_test.fit_marginal_from_joints()

    # make estimate
    signals = [r for r in inundation if r in bayes_network_t.signal_downward]
    infer_signal = [
        bayes_network_t.convert_state_to_dist(
            bayes_network_f.infer_node_states(s, 1, 1, 1)
        ) for s in signals
    ]
    marginals_flood, update_loc_f = bayes_network_t.update_network_with_multiple_soft_evidence(
        [
            {**{k: bayes_network_t.signal_downward[k]['speed_flood'] for k in signals},
             **{k: v for i in infer_signal if i[0] != {} for k, v in i[0].items()}},
            {**{k: bayes_network_t.signal_upward[k]['speed_flood'] for k in signals},
             **{k: v for i in infer_signal if i[1] != {} for k, v in i[1].items()}},
        ],
        [
            bayes_network_t.marginals_downward.copy(),
            bayes_network_t.marginals_upward.copy()
        ],
        bayes_network_t.joints.copy(),
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
        {k: v for k, v in bayes_network_t.marginals_downward.items() if k in update_locs},  # historical
        {k: v for k, v in bn_t_test.marginals_downward.items() if k in update_locs},  # ground truth
    ])

    kld.append([historical, estimated])
kld = [i for i in kld if i[0] is not None]
relative_changes = [(e1 - e2) / e1 if e1 != 0 else 0 for e1, e2 in kld]

print('End of program.')

