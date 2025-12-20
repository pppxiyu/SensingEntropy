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
import pickle
os.environ['PYTHONHASHSEED'] = str(0)
import pandas as pd


road_data = dd.RoadData()
road_data.load_instance(f'{dir_cache_instance}/road_data')
bayes_network_t = mo.TrafficBayesNetwork()
bayes_network_t.load_instance(f'{dir_cache_instance}/bayes_network_t')
bayes_network_t_normal = mo.TrafficBayesNetwork()
load_normal = bayes_network_t_normal.load_instance(f'{dir_cache_instance}/bayes_network_t_normal')
assert load_normal is True, "If load_normal fails, go to val_internal to run the codes"

# get sensing positioning plans
with open(f"{dir_results}/placement/sensing_1.pkl", "rb") as f:
    sensor_1 = pickle.load(f)
with open(f"{dir_results}/placement/sensors_2nd_n_all.pkl", "rb") as f:
    sensors_2nd_n_all = pickle.load(f)
    print(f"Loaded sensor info except for the 1st one from disk.")
sensors = [sensor_1['road']] + [i['road'] for i in sensors_2nd_n_all]

# validate on each incident
record_ict = []
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

    # This block is changed from internal validation script
    # estimate using the trained BN, get the long-term signals are inputs
    record_sensor_count = []
    for i in range(len(sensors)):
        random.seed(0)
        np.random.seed(0)
        df_random_seed = 0

        sensors_selected = sensors[:i + 1]

        inundation_observed = [k for k in inundation if k in sensors_selected]
        if inundation_observed:
            marginals_flood, update_loc_f, _, _ = bayes_network_t.update_network_with_multiple_soft_evidence(
                [{**{k: bayes_network_t.signal_downward[k]['speed_flood'] for k in inundation_observed}},  # signal_down
                {**{k: bayes_network_t.signal_upward[k]['speed_flood'] for k in inundation_observed}},],  # signal_up
                [bayes_network_t.marginals_downward.copy(), bayes_network_t.marginals_upward.copy()],  # marginals
                bayes_network_t.joints.copy(),  # joints
                verbose=0
            )

            update_locs = update_loc_f['down'] + update_loc_f['up']  # include signal locs

            # compared disruption level among layouts
            estimated_disruption = bayes_network_t.calculate_network_kl_divergence([
                {k: v for k, v in marginals_flood.items() if k in update_locs},  # estimate
                {k: v for k, v in bayes_network_t_normal.marginals.items() if k in update_locs},  # averaged (normal period)
            ], expand=True 
            )

            # compared unexpectedness (without flood belief) among layouts
            estimated_unexpectedness = bayes_network_t.calculate_network_kl_divergence([
                {k: v for k, v in marginals_flood.items() if k in update_locs},  # estimate
                {k: v for k, v in bayes_network_t.marginals_downward.items() if k in update_locs},  # averaged (flood period)
            ], expand=True  
        )
        else:
            update_loc_f = {'down': [], 'up': []}
            update_locs = update_loc_f['down'] + update_loc_f['up']  # include signal locs
            estimated_disruption, estimated_unexpectedness = [0], [0]

        # # eval
        print(f'Signal locs: {inundation_observed}')
        print(f'All traversed locs: {update_locs}')

        record = {
            'inundation': inundation,
            'sensor_count': i + 1,
            'sensors_selected': sensors_selected,
            'inundation_observed': inundation_observed,
            'update_locs': update_locs,
            'estimated_disruption': estimated_disruption,
            'estimated_unexpectedness': estimated_unexpectedness,
        }
        record_sensor_count.append(record)
    record_ict.append(record_sensor_count)

# # vis output
record_ict_2 = []
for ict in record_ict:
    all_sensors_dis = sum(ict[-1]['estimated_disruption'])
    all_sensors_unexp = sum(ict[-1]['estimated_unexpectedness'])
    if all_sensors_dis == 0:
        continue
    record_sensor_2 = []
    for s in ict:
        record_sensor_2.append(
            {
                'sensor_count': s['sensor_count'],
                'estimated_disruption_ratio': (
                    sum(s['estimated_disruption']) if s['estimated_disruption'] is not None else 0
                    ) / all_sensors_dis,
                'estimated_unexpectedness_ratio': (
                    sum(s['estimated_unexpectedness']) if s['estimated_unexpectedness'] is not None else 0
                    ) / all_sensors_unexp,
            }
        )
    record_ict_2.append(record_sensor_2)

sensor_n = len(record_ict_2[0])
averages = []
for n in range(sensor_n):
    dis_values = [i[n]['estimated_disruption_ratio'] for i in record_ict_2]
    unexp_values = [i[n]['estimated_unexpectedness_ratio'] for i in record_ict_2]
    ave = sum([
        i * weight_disruption + j * (1 - weight_disruption) for i, j in zip(dis_values, unexp_values)
    ]) / (len(dis_values))
    averages.append(ave)
print(averages)


print('End of program.')
