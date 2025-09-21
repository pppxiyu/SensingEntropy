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
    rd_test = dd.RoadData()  # empty instance
    inundation = rd_test.get_flooded_roads_during_inct(
        road_data.flood_time_per_road, # road time periods
        road_data.flood_time_citywide.copy().iloc[[i]]  # incident time period
    )

    # select incidents when inundations can enter bayes_network_t
    if len([k for k in inundation if k in bayes_network_t.marginals_downward]) != len(inundation):
        continue

    # get traffic data in the incident and resample
    d = rd_test.pull_nyc_dot_traffic_flooding(
        dir_NYC_data_token,
        select_incidents=road_data.flood_time_citywide.copy().iloc[[i]]
    )
    if d is None:
        continue
    rd_test.resample_nyc_dot_traffic(rd_test.speed)

    # fit marginals from the incident data
    bn_t_test = mo.TrafficBayesNetwork(
        speed=rd_test.speed_resampled, road_geo=road_data.geo,
        n_samples=10000, n_components=12,
        remove_nodes=remove_data_from_nodes, corr_thr=.5,
        # network_ready=bayes_network_t.network.copy()
    )
    bn_t_test.fit_marginal_from_data()

    # # select inundations can enter bn_t_test
    inundation = [k for k in inundation if k in bn_t_test.marginals]
    if inundation == []:
        continue

    # # using the real world signals to update BN
    # bn_t_test.fit_signal(
    #     road_data.flood_time_per_road, bayes_network_f.marginals,
    #     mode='from_data', upward=False,
    # )  # preparing for bn_t_test.convert_state_to_dist
    # inferred_inundation = [
    #     bayes_network_t.convert_state_to_dist(
    #         bayes_network_f.infer_node_states(s, 1, 1, 1)
    #     ) for s in inundation
    # ]
    # inferred_inundation = [bayes_network_f.infer_node_states(s, 1, 1, 1) for s in inundation]
    # inferred_inundation = [
    #     {k: v for k, v in i.items() if k in bn_t_test.marginals}
    #     for i in inferred_inundation
    # ]
    # if not all(item == {} for item in inferred_inundation):
    #     print('debug')

    # estimate
    marginals_flood, update_loc_f = bayes_network_t.update_network_with_multiple_soft_evidence(
        [
            {**{k: bn_t_test.marginals[k] for k in inundation},
            #  **{k: bn_t_test.marginals[k] for i in inferred_inundation if i != {} for k, v in i.items() if v == 'flooded'}
            },  # signal_down
            {**{k: bn_t_test.marginals[k] for k in inundation},
            #  **{k: bn_t_test.marginals[k] for i in inferred_inundation if i != {} for k, v in i.items() if v == 'flooded'}
            },  # signal_up
        ],
        [
            bayes_network_t.marginals_downward.copy(),
            bayes_network_t.marginals_upward.copy()
        ],  # marginals
        bayes_network_t.joints.copy(),  # joints
        verbose=0
    )

    # eval
    print(f'Signal locs: {inundation}')
    # update_locs = update_loc_f['down'] + update_loc_f['up']
    update_locs = [i for i in (update_loc_f['down'] +  update_loc_f['up']) if i not in inundation]  # remove signal locs
    print(f'All traversed locs: {update_locs}')

    estimated = bayes_network_t.calculate_network_kl_divergence([
        {k: v for k, v in marginals_flood.items() if k in update_locs},  # estimate
        {k: v for k, v in bn_t_test.marginals.items() if k in update_locs},  # incident ground truth
    ])
    historical = bayes_network_t.calculate_network_kl_divergence([
        {k: v for k, v in bayes_network_t.marginals_downward.items() if k in update_locs},  # averaged
        {k: v for k, v in bn_t_test.marginals.items() if k in update_locs},  # incident ground truth
    ])
    kld.append([historical, estimated])

kld = [i for i in kld if i[0] is not None]
relative_changes = [(e1 - e2) / e1 if e1 != 0 else 0 for e1, e2 in kld]

print('End of program.')

