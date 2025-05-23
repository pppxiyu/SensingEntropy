import numpy as np
import os
os.environ["OMP_NUM_THREADS"] = "1"
from sklearn.mixture import GaussianMixture
import visualization as vis


class TrafficBayesNetwork:
    def __init__(self, n_samples, max_components):
        self.network = None
        self.network_mode = None
        self.gmm_per_road = None
        self.gmm_joint_road_road = None
        self.gmm_joint_flood_road = None
        self.n_samples = n_samples
        self.max_components = max_components
        return

    def _optimal_gmm(self, x,):
        lowest_bic = np.inf
        best_gmm = None
        for n in range(1, self.max_components + 1):
            gmm = GaussianMixture(
                n_components=n,
                # covariance_type='diag',
                # init_params='k-means++',
                reg_covar=.2,
                n_init=3,
                random_state=66
            )
            gmm.fit(x)
            aic = gmm.aic(x)
            if aic < lowest_bic:
                lowest_bic = aic
                best_gmm = gmm
        return best_gmm

    def fit_marginal(self, df, ):
        segment_gmms = {}
        for segment, data in df.copy().groupby("link_id"):
            speeds = data["speed"].values.reshape(-1, 1)
            best_gmm = self._optimal_gmm(speeds,)
            segment_gmms[segment] = best_gmm
        self.gmm_per_road = segment_gmms
        return

    @staticmethod
    def _find_downstream_segment(segment, gdf):
        end_point = segment.geometry.coords[-1]
        downstream = []
        for _, other in gdf.iterrows():
            if segment["link_id"] == other["link_id"]:  # Skip self
                continue
            start_point = other.geometry.coords[0]
            if end_point == start_point:  # If they connect
                downstream.append(other["link_id"])
        return downstream

    def build_network_from_geo_legacy(self, roads_geo):
        from pgmpy.models import BayesianNetwork
        edges = []
        for _, row in roads_geo.iterrows():
            downstream_segments = self._find_downstream_segment(row, roads_geo)
            if downstream_segments:
                for downstream_segment in downstream_segments:
                    edges.append((downstream_segment, row["link_id"]))  # downstream affect upstream
        self.network = BayesianNetwork(edges)
        return

    def remove_no_data_segment_from_network(self, graph):
        node_list_graph = list(graph.nodes())
        node_list_gmm = list(self.gmm_per_road.keys())
        graph_node_not_in_gmm = []
        for n in node_list_graph:
            if n not in node_list_gmm:
                graph_node_not_in_gmm.append(n)
        if not graph_node_not_in_gmm:
            return graph
        else:
            for nn in graph_node_not_in_gmm:
                predecessors = list(graph.predecessors(nn))
                successors = list(graph.successors(nn))
                for pred in predecessors:
                    for succ in successors:
                        graph.add_edge(pred, succ)
                graph.remove_node(nn)

            graph_covered_by_speed_data = True
            for n in list(graph.nodes()):
                if n not in node_list_gmm:
                    graph_covered_by_speed_data = False
                    break
            assert graph_covered_by_speed_data, 'Some nodes in the network do not have speed data.'
            return graph

    def build_network_from_geo(self, gdf, mode='causal', remove_no_data_segment=True):
        import networkx as nx
        graph = nx.DiGraph()

        segment_groups = {}
        for _, row in gdf.iterrows():
            link_id = row['link_id']
            geometry = row['geometry']
            if link_id not in segment_groups:
                segment_groups[link_id] = []
            segment_groups[link_id].append({
                'start': geometry.coords[0],
                'end': geometry.coords[-1],
                'coords': list(geometry.coords),
                'geometry': geometry
            })  # some segments were split into two

        for link_id, segments in segment_groups.items():
            graph.add_node(link_id, segments=segments)

        tolerance = 1e-6
        for source_link_id, source_segments in segment_groups.items():
            for target_link_id, target_segments in segment_groups.items():
                if source_link_id == target_link_id:
                    continue

                for source_seg in source_segments:
                    source_end = source_seg['end']
                    for target_seg in target_segments:
                        target_coords = target_seg['coords']

                        for coord in target_coords[:-1]:  # exclude end point of the target
                            if (abs(source_end[0] - coord[0]) < tolerance and
                                    abs(source_end[1] - coord[1]) < tolerance):
                                graph.add_edge(source_link_id, target_link_id)
                                break

        if not nx.is_directed_acyclic_graph(graph):
            import warnings
            warnings.warn('The network is not acyclic.')
        if remove_no_data_segment:
            graph = self.remove_no_data_segment_from_network(graph)

        if mode == 'causal':
            graph = graph.reverse(copy=True)
            self.network_mode = 'causal'
        else:
            assert mode == 'physical'
            self.network_mode = 'physical'
            import warnings
            warnings.warn('The edge directions follows traffic flow. It should be reverse for Bayesian network. ')

        self.network = graph
        return

    def build_network_by_causality(
            self, road_speed, road_geo, local_crs,
            significance=0.01, max_lag=2, max_upstream_roads=2
    ):
        # legacy call in main: build_network_by_causality(
        #     road_data.speed_resampled, road_data.geo, local_crs,
        #     significance=0.05, max_lag=1, max_upstream_roads=1
        # )
        from statsmodels.tsa.stattools import grangercausalitytests
        from collections import defaultdict
        from pgmpy.models import BayesianNetwork

        road_speed = road_speed.pivot(index="time", columns="link_id", values="speed")
        causality_scores = {}
        edges = []
        examined_pairs = set()

        road_geo = road_geo[road_geo["link_id"].isin(road_speed.columns)]
        road_geo = road_geo.to_crs(local_crs)
        road_geo["x"] = road_geo.geometry.centroid.x
        road_geo["y"] = road_geo.geometry.centroid.y

        for seg_i in road_speed.columns:
            nearby_segments = self._get_nearby_segments(road_geo, seg_i,)
            for seg_j in nearby_segments:
                if (seg_i, seg_j) in examined_pairs or (seg_j, seg_i) in examined_pairs:
                    continue
                examined_pairs.add((seg_i, seg_j))

                result_ab = grangercausalitytests(road_speed[[seg_i, seg_j]].dropna(), max_lag)
                p_value_ab = min(result_ab[lag][0]['ssr_ftest'][1] for lag in range(1, max_lag + 1))
                result_ba = grangercausalitytests(road_speed[[seg_j, seg_i]].dropna(), max_lag,)
                p_value_ba = min(result_ba[lag][0]['ssr_ftest'][1] for lag in range(1, max_lag + 1))

                if p_value_ab < significance and p_value_ba < significance:
                    if p_value_ab < p_value_ba:
                        causality_scores[(seg_i, seg_j)] = p_value_ab
                    else:
                        causality_scores[(seg_j, seg_i)] = p_value_ba
                elif p_value_ab < significance:
                    causality_scores[(seg_i, seg_j)] = p_value_ab
                elif p_value_ba < significance:
                    causality_scores[(seg_j, seg_i)] = p_value_ba

            upstream_roads = defaultdict(list)
            for (i, j), p_value in sorted(causality_scores.items(), key=lambda x: x[1]):
                upstream_roads[j].append((i, p_value))
            for downstream, candidates in upstream_roads.items():
                top_candidates = sorted(candidates, key=lambda x: x[1])[:max_upstream_roads]
                for upstream, _ in top_candidates:
                    edges.append((upstream, downstream))

        edges = self._remove_cycles(edges)
        self.network = BayesianNetwork(edges)
        return

    @staticmethod
    def _get_nearby_segments(road_geo, seg_i, buffer_distance=4000):
        from shapely.geometry import Point
        if seg_i not in road_geo['link_id'].to_list():
            return [i for i in road_geo['link_id'].to_list() if i != seg_i]

        seg_i_point = road_geo.loc[road_geo['link_id'] == seg_i, ['x', 'y']].values[0]
        seg_i_geom = Point(seg_i_point)
        s_index = road_geo.sindex
        possible_matches = list(s_index.intersection(seg_i_geom.buffer(buffer_distance).bounds))
        nearby_segments = road_geo.iloc[possible_matches].copy()
        nearby_segments['distance'] = nearby_segments.apply(
            lambda row: seg_i_geom.distance(Point(row['x'], row['y'])), axis=1
        )
        nearby_segments = nearby_segments[nearby_segments['distance'] <= buffer_distance]
        return nearby_segments.loc[nearby_segments['link_id'] != seg_i, 'link_id'].tolist()

    @staticmethod
    def _remove_cycles(edges):
        import networkx as nx
        graph = nx.DiGraph(edges)
        while not nx.is_directed_acyclic_graph(graph):
            cycles = list(nx.simple_cycles(graph))
            if not cycles:
                break
            cycle = cycles[0]
            edge_to_remove = (cycle[-1], cycle[0])
            print(f"Removing cycle edge: {edge_to_remove}")
            graph.remove_edge(*edge_to_remove)
        return list(graph.edges())

    def fit_joint(self, df, ):
        dependency_gmms = {}
        i = 0
        for child in self.network.nodes():
            i += 1
            parents = list(self.network.predecessors(child))
            if not parents:
                continue

            parents_filtered = []
            for parent in parents:
                if parent in df['link_id'].unique():
                    parents_filtered.append(parent)

            parent_data = df[df['link_id'].isin(parents)].pivot(
                index='time', columns='link_id', values='speed'
            ).dropna()
            child_data = df[df['link_id'] == child].set_index('time')["speed"].dropna()

            parent_data = parent_data.reindex(child_data.index).dropna()
            child_data = child_data.reindex(parent_data.index).dropna()

            if len(parent_data) == 0 or len(child_data) == 0:
                continue

            x = child_data.values.reshape(-1, 1)
            y = parent_data.values  # keep the child at the first dim

            gmm = GaussianMixture(
                n_components=self.max_components,
                random_state=66,
                reg_covar=.2,
                n_init=3,
            )
            gmm.fit(np.hstack((x, y)))
            dependency_gmms[child] = {'parents': parents_filtered, 'joints': gmm, 'parent_type': self.network_mode}
        self.gmm_joint_road_road = dependency_gmms
        return

    def fit_joint_flood_n_speed(self, flood_time, speed, flood_prob, verbose=0):
        import warnings

        dists = {}
        for _, row in flood_prob.iterrows():
            link_id = row['link_id']
            p_flood = row['p']

            speed_data = speed[speed['link_id'] == link_id].copy()
            if speed_data.empty or link_id not in flood_time:
                dists[link_id] = {
                    'p_flood': p_flood,
                    'speed_no_flood': None,
                    'speed_flood': None
                }
                continue

            flood_events = flood_time[link_id]
            speed_data['flooded'] = False
            for _, event in flood_events.iterrows():
                mask = (speed_data['time'] >= event['buffer_start']) & (speed_data['time'] <= event['buffer_end'])
                speed_data.loc[mask, 'flooded'] = True
            speed_flood = speed_data[speed_data['flooded']]['speed']
            speed_no_flood = speed_data[~speed_data['flooded']]['speed']

            def get_gmm(df):
                if (len(df)) > 50 and (df.sum() != 0):
                    gmm = self._optimal_gmm(df.values.reshape(-1, 1),)
                else:
                    gmm = None
                    if verbose > 0:
                        warnings.warn(f'Sample size is too small. Skipped.')
                return gmm

            gmm_no_flood = get_gmm(speed_no_flood)
            gmm_flood = get_gmm(speed_flood)

            dists[link_id] = {
                'p_flood': p_flood,
                'speed_no_flood': gmm_no_flood,
                'speed_flood': gmm_flood
            }
        self.gmm_joint_flood_road = dists
        return

    def calculate_gmm_entropy_approximation(self, gmm):
        samples, _ = gmm.sample(self.n_samples)
        log_probs = gmm.score_samples(samples)
        entropy = -np.mean(log_probs)
        return entropy

    def calculate_network_entropy(self, marginals=None, joints=None, verbose=1):
        import networkx as nx
        # if the node has no parent, the marginal was used
        # if the node has parents, joint and the marginal refitted from joint sampling was used

        if marginals is None:
            marginals = self.gmm_per_road
        if joints is None:
            joints = self.gmm_joint_road_road

        topo_order = list(nx.topological_sort(self.network))
        total_entropy = 0.0
        for node in topo_order:
            parents = list(self.network.predecessors(node))
            if not parents:
                gmm = marginals[node]
                entropy = self.calculate_gmm_entropy_approximation(gmm)
                if verbose > 1:
                    print(f"Node {node} (root): H({node}) = {entropy:.4f}")

            else:
                assert node in joints, 'The node does not have parent.'
                parent_ids, joint_gmm = joints[node]['parents'], joints[node]['joints']
                h_joint = self.calculate_gmm_entropy_approximation(joint_gmm)

                joint_samples = joint_gmm.sample(self.n_samples)[0]  # NOTE: time-consuming!!
                parent_samples = joint_samples[:, 1:]
                h_parents = self.calculate_gmm_entropy_approximation(self._optimal_gmm(parent_samples))

                entropy = h_joint - h_parents
                if verbose > 1:
                    print(f"Node {node}: H({node}|{parents}) = {entropy:.4f}")
            total_entropy += entropy
        if verbose >= 1:
            print(f"Total Entropy of the Bayesian Network: {total_entropy}")
        return total_entropy

    def update_joints(
            self, orig_joint, orig_xk_gmm, observed_xk_gmm, link_name, parent_ids,
    ):
        # sample from original joint, p(A, B_1, B_2), for example
        joint_samples = orig_joint.sample(self.n_samples)[0]
        xk_idx = parent_ids.index(link_name) + 1  # +1 because child is first
        xk_samples = joint_samples[:, [xk_idx]]

        # get the ratio term, p'(B_1)/p(B_1)
        log_weights = observed_xk_gmm.score_samples(xk_samples) - orig_xk_gmm.score_samples(xk_samples)
        weights = np.exp(log_weights - np.max(log_weights))
        weights /= weights.sum()

        # apply the ratio term
        indices = np.random.choice(self.n_samples, size=self.n_samples, p=weights, replace=True)
        return joint_samples[indices]

    def update_joint_at_child_legacy(
            self, orig_joint, orig_child_gmm, observed_child_gmm,
    ):
        # it is actually the same as refitting the joint
        # the advantage is the non-updated dimensions remain the same with surety

        # sample from original joint, p(A, B_1, B_2), for example
        joint_samples = orig_joint.sample(self.n_samples)[0]
        child_samples = joint_samples[:, [0]]

        # get the ratio term, p'(B_1)/p(B_1)
        log_weights = (observed_child_gmm.score_samples(child_samples)
                       - orig_child_gmm.score_samples(child_samples))
        weights = np.exp(log_weights - np.max(log_weights))
        weights /= weights.sum()

        # apply the ratio term
        indices = np.random.choice(self.n_samples, size=self.n_samples, p=weights, replace=True)
        return joint_samples[indices]

    def update_network_with_soft_evidence(
            self, link_name, observed_marginal, marginals, joints, verbose=1
    ):
        import copy
        from queue import Queue
        marginals_updated = copy.deepcopy(marginals)
        marginals_fixed = copy.deepcopy(marginals)
        joints_updated = copy.deepcopy(joints)

        if observed_marginal is None:
            return marginals_updated, joints_updated

        queue = Queue()
        queue.put(link_name)
        processed = set()
        marginals_updated[link_name] = observed_marginal

        while not queue.empty():
            current_node = queue.get()
            if current_node in processed:
                continue
            processed.add(current_node)

            for child in self.network.successors(current_node):
                assert child in joints_updated
                parent_ids, joint_gmm = joints_updated[child]['parents'], joints_updated[child]['joints']
                assert current_node in parent_ids

                # update joint
                resampled_joint_samples = self.update_joints(
                    joint_gmm, marginals_fixed[current_node], marginals_updated[current_node],
                    current_node, parent_ids,
                )
                new_joint_gmm = self._optimal_gmm(resampled_joint_samples,)
                joints_updated[child] = {'parents': parent_ids, 'joints': new_joint_gmm}
                if verbose > 0:
                    print(f"Updated joint for {child} with parents {parent_ids}")

                # update marginal
                resampled_child_samples = resampled_joint_samples[:, [0]]
                new_child_gmm = self._optimal_gmm(resampled_child_samples,)
                marginals_updated[child] = new_child_gmm
                if verbose > 0:
                    print(f"Updated marginal for {child}")

                queue.put(child)

        return marginals_updated, joints_updated

    def update_network_with_soft_evidence_2(
            self, link_name, observed_marginal, marginals, joints, verbose=1
    ):
        import copy
        from queue import Queue
        marginals_updated = copy.deepcopy(marginals)
        marginals_fixed = copy.deepcopy(marginals)
        joints_updated = copy.deepcopy(joints)

        if observed_marginal is None:
            return marginals_updated, joints_updated

        queue = Queue()
        queue.put(link_name)
        processed = set()
        marginals_updated[link_name] = observed_marginal

        while not queue.empty():
            current_node = queue.get()
            if current_node in processed:
                continue
            processed.add(current_node)

            for child in self.network.successors(current_node):
                assert child in joints_updated
                parent_ids, joint_gmm = joints_updated[child]['parents'], joints_updated[child]['joints']
                assert current_node in parent_ids

                # update joint
                resampled_joint_samples = self.update_joints(
                    joint_gmm, marginals_fixed[current_node], marginals_updated[current_node],
                    current_node, parent_ids,
                )
                new_joint_gmm = self._optimal_gmm(resampled_joint_samples,)
                joints_updated[child] = {'parents': parent_ids, 'joints': new_joint_gmm}
                if verbose > 0:
                    print(f"Updated joint for {child} with parents {parent_ids}")

                # update marginal
                resampled_child_samples = resampled_joint_samples[:, [0]]
                new_child_gmm = self._optimal_gmm(resampled_child_samples,)
                marginals_updated[child] = new_child_gmm
                if verbose > 0:
                    print(f"Updated marginal for {child}")

                queue.put(child)

        return marginals_updated, joints_updated

    def calculate_network_conditional_entropy(self, network_list, verbose=1):
        # Conditional entropy: H(A∣B)=∑_b P(B=b)H(A∣B=b)
        # Example: 0.01 * entropy {network updated with flood time dist_w_obs}
        # + 0.99 * entropy {network with non-flood time dist_w_obs}
        entropy = 0
        for n in network_list:
            entropy += n['p'] * self.calculate_network_entropy(
                marginals=n['marginals'], joints=n['joints'], verbose=0,
            )
        if verbose > 0:
            print(f"Total Entropy of the Bayesian Network: {entropy}")
        return entropy

    def get_entropy_with_signals(self, bayes_joints, bayes_marginal, signal_dict, if_vis=False):
        entropies = {}
        for k, v in signal_dict.items():
            if (v['speed_no_flood'] is not None) and (v['speed_flood'] is not None):
                marginals_no_flood, joints_no_flood = self.update_network_with_soft_evidence(
                    k, v['speed_no_flood'], bayes_marginal, bayes_joints, verbose=0
                )
                marginals_flood, joints_flood = self.update_network_with_soft_evidence(
                    k, v['speed_flood'], bayes_marginal, bayes_joints, verbose=0
                )
                entropies[k] = self.calculate_network_conditional_entropy([
                    {'p': 1 - v['p_flood'], 'marginals': marginals_no_flood, 'joints': joints_no_flood},
                    {'p': v['p_flood'], 'marginals': marginals_flood, 'joints': joints_flood},
                ])

                if if_vis:
                    if (k in joints_flood.keys()) and (len(joints_flood[k][0]) == 1):
                        z_max = vis.dist_gmm_3d(joints_flood[k][1], k, joints_flood[k][0][0], return_z_max=True)
                        vis.dist_gmm_3d(bayes_joints[k][1], k, bayes_joints[k][0][0], z_limit=z_max
                        )  # FIGURE 4: joint distributions changes
        return entropies


class FloodBayesNetwork:
    def __init__(self, t_window='D'):
        """
        :param t_window: the unit for processing flood data, default is day
        """
        self.t_window = t_window
        self.network = None
        self.marginals = None
        self.conditionals = None
        self.network_bayes = None

    def fit_marginal(self, df, if_vis=False):
        """
        df should contain time_create, id, and link_id col
        time_create col is the start time of the road closure
        link_id col is the id of the road
        id col is the id of the closure
        """
        from pandas.api.types import is_datetime64_ns_dtype
        from pandas.api.types import is_string_dtype
        assert 'time_create' in df.columns, 'Column time_create not found in the DataFrame'
        assert is_datetime64_ns_dtype(df['time_create']), 'Column time_create is not in type datetime64[ns]'
        assert 'link_id' in df.columns, 'Column link_id not found in the DataFrame'
        assert is_string_dtype(df['link_id']), 'Column link_id is not of string type'
        assert 'id' in df.columns, 'Column id not found in the DataFrame'
        assert is_string_dtype(df['id']), 'Column id is not of string type'

        if if_vis:
            vis.bar_closure_count_over_time(df.copy())
        pass

        df.loc[:, 'time_bin'] = df['time_create'].dt.floor(self.t_window)
        df = df.drop_duplicates(subset=['link_id', 'time_bin'])

        flooding_day_count = df['time_bin'].nunique()
        closure_count = df.groupby('link_id').count()['id']

        closure_count_p = closure_count / flooding_day_count
        closure_count_p = closure_count_p.reset_index().rename(columns={'id': 'p'})
        self.marginals = closure_count_p
        return

    def build_network_by_co_occurrence(self, df, weight_thr=0, report=False):
        """
        df should contain time_create and link_id col
        time_create col is the start time of the road closure
        link_id col is the id of the road

        weight_thr: edges with weights below the threshold will be removed

        Note that the network topology is not perfectly defined. Please consider the facts below:

        Co-occurrence network is used to define the topology, and it is problematic when the
        occurrence of flooding is rare because of the lack of samples.

        The algo below would produce double connections between two nodes, e.g., A to B and B to A,
        so the connection with lower weight was removed.

        The temporal dimension is not considered. B happens after A (not just co-occurrence) has more
        useful information.

        Considering the elevation of roads could be helpful.
        """

        import networkx as nx

        _, occurrence, co_occurrence = self.process_raw_flood_data(df.copy())

        graph = nx.DiGraph()
        graph.add_nodes_from(df.copy()['link_id'].unique())
        for node in graph.nodes:
            graph.nodes[node]['occurrence'] = occurrence[node]
        for (a, b), count in co_occurrence.items():
            prob = count / occurrence[a]
            graph.add_edge(a, b, weight=prob)

        graph, _ = self.remove_min_weight_feedback_arcs(graph)

        assert isinstance(weight_thr, float), "weight_thr should be a float"
        edges_to_remove = [
            (u, v) for u, v, d in graph.edges(data=True)
            if d.get('weight', 0) < weight_thr
        ]
        graph.remove_edges_from(edges_to_remove)

        if report:
            for node, indegree in graph.in_degree():
                print(f"Node {node}: in-degree = {indegree}")
        self.network = graph
        return

    def fit_conditional(self, df):
        """
        df should contain time_create and link_id col
        time_create col is the start time of the road closure
        link_id col is the id of the road
        """

        from collections import defaultdict
        from itertools import product

        time_groups, _, _ = self.process_raw_flood_data(df.copy())
        conditionals = defaultdict(dict)

        for node in self.network.nodes:
            parents = list(self.network.predecessors(node))
            if len(parents) == 0:
                continue
            if len(parents) > 4:
                import warnings
                warnings.warn(
                    f"""
                    Time complexity of this step is 2^n. n = {len(parents)} right now.
                    Larger n could also potentially cause the sparsity issue.
                    Try tuning the weight threshold to mitigate this issue. 
                    """
                )

            parent_states = list(product([0, 1], repeat=len(parents)))
            counts = {state: {'co-occur': 0, 'occur': 0} for state in parent_states}

            for _, flood_road in time_groups.items():
                # check which parent state does the flood incident go to
                for state in parent_states:
                    parent_state_matches = all(
                        (p in flood_road) == bool(s)  # if p is flooded matches state under checking
                        for p, s in zip(parents, state)
                    )
                    if parent_state_matches:
                        counts[state]['occur'] += 1
                        if node in flood_road:
                            counts[state]['co-occur'] += 1

            assert sum([v['occur'] for k, v in counts.items()]) == len(time_groups), 'Count inconsistent.'
            assert (sum([v['co-occur'] for k, v in counts.items()]) / len(time_groups)
                    == self.marginals.loc[self.marginals['link_id'] == node, 'p'].values), 'Count inconsistent.'

            cond_probs = {}
            for k, v in counts.items():
                parents_occur_count = v['occur']
                co_occur_count = v['co-occur']
                if parents_occur_count == 0:
                    p = 0
                else:
                    p = co_occur_count / parents_occur_count
                cond_probs[k] = p

            conditionals[node] = {
                'parents': parents,
                'conditionals': cond_probs
            }

        self.conditionals = conditionals
        return

    def build_bayes_network(self):
        from pgmpy.models import BayesianNetwork
        from pgmpy.factors.discrete import TabularCPD
        from itertools import product

        edges = list(self.network.edges())
        network_bayes = BayesianNetwork(edges)

        for node in self.network.nodes:  # add info for nodes w/o parent
            if self.network.degree(node) == 0:
                continue

            if self.network.in_degree(node) == 0:
                p_flood = self.marginals.loc[self.marginals['link_id'] == node, 'p'].values[0]
                mpd = TabularCPD(
                    variable=node,
                    variable_card=2,
                    values=[[1 - p_flood], [p_flood]],
                )
                network_bayes.add_cpds(mpd)

            else:  # add info for nodes w parent
                parents = self.conditionals[node]['parents']
                parent_states = list(product([0, 1], repeat=len(parents)))
                p1 = [self.conditionals[node]['conditionals'].get(state) for state in parent_states]
                p0 = [1 - p for p in p1]

                cpd = TabularCPD(
                    variable=node,
                    variable_card=2,
                    evidence=parents,
                    evidence_card=[2] * len(parents),
                    values=[p0, p1],
                )
                network_bayes.add_cpds(cpd)

        self.network_bayes = network_bayes
        return

    def check_bayesian_network(self):
        """
        Check if the inferred p of flooding for nodes is consistent with the p calculated as marginals.
        """
        from pgmpy.inference import VariableElimination

        inference = VariableElimination(self.network_bayes)
        for node in self.network_bayes.nodes():
            if self.network_bayes.in_degree(node) == 0:
                continue
            p = inference.query(variables=[node]).values

            if self.marginals.loc[self.marginals['link_id'] == node, 'p'].values[0] != p[1]:
                import warnings
                warnings.warn(
                    """
                        Probability in Bayesian Network and fitted marginals should be the same 
                        before any observations. There could be small differences becasue of numerical issue.
                        Check that manually if they are not the same.
                    """
                )
                print(self.marginals.loc[self.marginals['link_id'] == node, 'p'].values[0], p[1])

    def infer_w_evidence(self, target_node, evidence):
        """
        Get the probability of flooding for roads and return 'flooded roads' with a threshold.
        Example input:
            target_node = 'A'
            evidence = {'B': 1, 'C': 0}, where 1 = flooded, 0 = not flooded
        """
        from pgmpy.inference import VariableElimination
        assert isinstance(target_node, str), "target_node must be a string."
        assert target_node in self.network_bayes.nodes, f"{target_node} is not in the network."
        assert isinstance(evidence, dict), "evidence must be a dictionary."
        for node, state in evidence.items():
            assert isinstance(node, str), f"Evidence key '{node}' must be a string."
            assert node in self.network_bayes.nodes, f"Evidence node '{node}' is not in the network."
            assert state in [0, 1], f"Evidence value for '{node}' must be 0 or 1, got {state}."

        inference = VariableElimination(self.network_bayes)
        result = inference.query(variables=[target_node], evidence=evidence)
        p = result.values
        return p

    @staticmethod
    def remove_min_weight_feedback_arcs(graph):
        import networkx as nx

        graph = graph.copy()
        removed_edges = []

        while not nx.is_directed_acyclic_graph(graph):
            try:
                cycle = next(nx.simple_cycles(graph))
            except StopIteration:
                break

            min_weight = float('inf')
            min_edge = None

            for i in range(len(cycle)):
                u, v = cycle[i], cycle[(i + 1) % len(cycle)]
                w = graph[u][v].get('weight',)
                if w < min_weight:
                    min_weight = w
                    min_edge = (u, v)

            graph.remove_edge(*min_edge)
            removed_edges.append(min_edge)

        return graph, removed_edges

    def process_raw_flood_data(self, df):
        from pandas.api.types import is_datetime64_ns_dtype
        from pandas.api.types import is_string_dtype
        from collections import defaultdict
        from itertools import permutations

        assert 'time_create' in df.columns, 'Column time_create not found in the DataFrame'
        assert is_datetime64_ns_dtype(df['time_create']), 'Column time_create is not in type datetime64[ns]'
        assert 'link_id' in df.columns, 'Column link_id not found in the DataFrame'
        assert is_string_dtype(df['link_id']), 'Column link_id is not of string type'

        df.loc[:, 'time_bin'] = df['time_create'].dt.floor(self.t_window)
        time_groups = df.groupby('time_bin')['link_id'].apply(set)
        occurrence = defaultdict(int)
        co_occurrence = defaultdict(int)
        for links in time_groups:
            for a in links:
                occurrence[a] += 1
            for a, b in permutations(links, 2):
                co_occurrence[(a, b)] += 1

        return time_groups, occurrence, co_occurrence





















