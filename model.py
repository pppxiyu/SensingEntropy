import warnings

import os
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import geopandas as gpd

from sklearn.mixture import GaussianMixture
import visualization as vis
import networkx as nx


class TrafficBayesNetwork:
    def __init__(
            self,
            speed: pd.DataFrame = None, road_geo: gpd.GeoDataFrame = None,
            n_samples: int = None, n_components: int = None,
            network_mode: str = 'causal', fitting_mode: str = 'one-off',
            remove_nodes: list = None, corr_thr: float = None, 
    ):
        """
        network_mode: it specifies the direction of edges in the network. "causal" means the arrow points
            from downstream nodes to upstream nodes, following the direction of traffic congestion propagation.
            "physical" means following the direction of traffic flows.
        speed: this arg is also used to remove edges with no data when building the network.
        align_data: this func is used to trim the data to ensure that data is consistent throughout
            fitting joints, marginals, and signals
        """
        assert fitting_mode in ['one-off', 'iterate']
        assert network_mode in ['causal', 'physical']

        self.network = None
        self.speed_data = None
        self.speed_data_raw = None
        self.marginals = None
        self.marginals_downward = None
        self.marginals_upward = None
        self.joints = None
        self.signal_downward = None
        self.signal_upward = None

        self.network_mode = network_mode
        self.n_samples = n_samples
        self.n_components = n_components
        self.fitting_mode = fitting_mode

        if speed is not None and road_geo is not None:
            self.speed_data_raw = speed
            self.network = self.build_network_from_geo(
                road_geo, remove_no_data_segment=speed, correlation_threshold=corr_thr, speed_df=speed
            )

            """
                In the Bayesian Network, the marginal of the same node should be similar (if not identical)
                among different joints. So, the raw data should be filtered by timestamp. Only timestamps
                that are available at all nodes are kept. In light of this, removing some nodes with
                very limited data is beneficial for maintaining the size of data. Use the function below 
                in the bebug mode to identify these harmful nodes using the greedy algorithm. 
                Then, input the results in the config.py file.
                
                self.remove_nodes_for_consistency(
                    speed.copy(), self.network.copy(), 5
                )
            """
            if remove_nodes is not None:
                speed = speed[~speed['link_id'].isin(remove_nodes)]
                self.remove_nodes_from_network(remove_nodes)

            self.speed_data = self.align_data_time(speed, self.network, check_kept_data=False)

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

    @staticmethod
    def remove_no_data_segment_from_network(graph, data):
        """
        Removes nodes from the graph that do not have corresponding data in the DataFrame.
        For each such node, connects all its predecessors to all its successors before removal to preserve possible paths.

        Parameters:
            graph (networkx.DiGraph): The network graph whose nodes correspond to 'link_id' values.
            data (pd.DataFrame): DataFrame containing at least a 'link_id' column.

        Returns:
            networkx.DiGraph: The modified graph with only nodes that have data.
        """
        
        # Get a list of all nodes in the graph
        node_list_graph = list(graph.nodes())
        # Get a list of all unique link_ids present in the data DataFrame
        node_list_gmm = list(data['link_id'].unique())
        graph_node_not_in_gmm = []
        # For each node in the graph, check if it is missing from the data
        for n in node_list_graph:
            if n not in node_list_gmm:
                graph_node_not_in_gmm.append(n)

        if not graph_node_not_in_gmm:
            return graph
        else:
            # For each node missing from the data
            for nn in graph_node_not_in_gmm:
                # Get all predecessors (incoming nodes) of this node
                predecessors = list(graph.predecessors(nn))
                # Get all successors (outgoing nodes) of this node
                successors = list(graph.successors(nn))
                # For every predecessor and every successor, add an edge between them
                for pred in predecessors:
                    for succ in successors:
                        graph.add_edge(pred, succ)
                graph.remove_node(nn)

            # After removal, check that all remaining nodes in the graph are present in the data
            graph_covered_by_speed_data = True
            for n in list(graph.nodes()):
                if n not in node_list_gmm:
                    graph_covered_by_speed_data = False
                    break
            # Assert that all nodes in the graph now have data; otherwise, raise an error
            assert graph_covered_by_speed_data, 'Some nodes in the network do not have speed data.'
            return graph

    @staticmethod
    def if_edges_correlation_over_thr(source, target, speed_df, threshold,versbose):
        # Extract speed time series for source and target
        source_df = speed_df[speed_df['link_id'] == source][['time', 'speed']].rename(columns={'speed': 'source_speed'})
        target_df = speed_df[speed_df['link_id'] == target][['time', 'speed']].rename(columns={'speed': 'target_speed'})

        # Merge on time to align the two series
        merged = pd.merge(source_df, target_df, on='time')

        # Calculate correlation (drop NaNs automatically)
        if not merged.empty:
            corr = merged['source_speed'].corr(merged['target_speed'])
            if versbose > 0:
                print(f"Correlation between {source} and {target}: {corr}")
        else:
            warnings.warn(f"No overlapping time data between {source} and {target}.")
            return False
        return corr > threshold
        

    def build_network_from_geo(
            self, gdf, remove_no_data_segment, remove_isolated_nodes=True,
            correlation_threshold=None, speed_df=None, verbose=0,
        ):
        """
            remove_no_data_segment should be a dataframe contains the data of segments
        """

        assert not (correlation_threshold is not None and speed_df is None)

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
                                
                                if correlation_threshold is not None:  # remove edges with low correlation
                                    if self.if_edges_correlation_over_thr(
                                        source_link_id, target_link_id, speed_df, correlation_threshold, verbose
                                    ):
                                        graph.add_edge(source_link_id, target_link_id)
                                else:
                                    graph.add_edge(source_link_id, target_link_id)
                                break

        assert nx.is_directed_acyclic_graph(graph), 'The network is not acyclic.'

        graph = self.remove_no_data_segment_from_network(graph, remove_no_data_segment)
        # Remove edges from the graph if their correlation is below the threshold
        if correlation_threshold is not None and speed_df is not None:
            edges_to_remove = []
            for source, target in list(graph.edges()):
                if not self.if_edges_correlation_over_thr(source, target, speed_df, correlation_threshold, verbose):
                    edges_to_remove.append((source, target))
            graph.remove_edges_from(edges_to_remove)

        if self.network_mode == 'causal':
            graph = graph.reverse(copy=True)
        else:
            import warnings
            warnings.warn('The edge directions follows traffic flow. It should be upward for Bayesian network. ')

        if remove_isolated_nodes:
            graph.remove_nodes_from(list(nx.isolates(graph)))

        return graph

    def remove_nodes_from_network(self, remove_nodes):
        self.network.remove_nodes_from(remove_nodes)
        self.network.remove_nodes_from(list(nx.isolates(self.network)))
        return

    def build_network_by_causality_legacy(
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

    def align_data_time(self, df, graph, check_kept_data=False,):
        """
        Aligns time series data across all nodes in the network for each connected component,
        so that only timestamps present for every node in each connected component are kept.

        Parameters:
            df (pd.DataFrame): Input DataFrame with at least 'link_id' and 'time' columns.
            graph (networkx.Graph or DiGraph): The network graph whose nodes correspond to 'link_id' values.
            check_kept_data (bool): If True, also returns statistics about data loss per node due to alignment.

        Returns:
            pd.DataFrame: Filtered DataFrame containing only rows with timestamps present for all nodes in each component.
            (optional) float: If check_kept_data is True, also returns the average loss ratio of data per node.
        """

        # For each node, get the set of timestamps where it has data
        timestamps_per_node = (
            df.groupby('link_id')['time']
            .apply(set)
            .to_dict()
        )

        dfs_consistent = []

        # Find connected components (weakly for directed graphs)
        components = (
            nx.connected_components(graph) if not graph.is_directed()
            else nx.weakly_connected_components(graph)
        )
        for component in components:
            # Find timestamps present for all nodes in the component
            common_timestamps = set.intersection(
                *[timestamps_per_node.get(node, set()) for node in component]
            )
            # Filter df to only rows for nodes in this component
            df_component = df[df['link_id'].isin(component)]
            # Further filter to only rows with common timestamps
            df_component = df_component[df_component['time'].isin(common_timestamps)]
            dfs_consistent.append(df_component)
        df_final = pd.concat(dfs_consistent, ignore_index=True)

        if check_kept_data:
            original_counts = df.groupby('link_id').size().rename('original_count')
            kept_counts = df_final.groupby('link_id').size().rename('kept_count')
            stats = original_counts.to_frame().join(kept_counts, how='left').fillna(0).astype(int)
            stats['lost_count'] = stats['original_count'] - stats['kept_count']
            stats['loss_ratio'] = stats['lost_count'] / stats['original_count']
            ave_loss_ratio = stats['loss_ratio'].mean()
            return df_final, ave_loss_ratio

        return df_final

    def evaluate_node_removal_impact(self, df, graph):
        baseline_df_final, ave_loss_r = self.align_data_time(df, graph, check_kept_data=True)

        results = []
        for node in graph.nodes():
            graph_copy = graph.copy()
            graph_copy.remove_node(node)

            df_filtered = df[df['link_id'] != node]
            df_final_after_removal, ave_loss_r_remove = self.align_data_time(
                df_filtered, graph_copy, check_kept_data=True
            )

            improvement = ave_loss_r - ave_loss_r_remove

            results.append({
                'link_id': node,
                'retained_with_removal': ave_loss_r_remove,
                'baseline_retained': ave_loss_r,
                'improvement': improvement
            })

        result_df = pd.DataFrame(results)
        result_df = result_df.sort_values(by='improvement', ascending=False)
        return result_df

    def remove_nodes_for_consistency(self, df, G, num_node_2_remove):
        graph = G.copy()
        remove_list = []
        for _ in range(num_node_2_remove):
            result_df = self.evaluate_node_removal_impact(df, graph)
            remove_node = result_df.iloc[0]['link_id']
            remove_list.append(remove_node)

            graph.remove_node(remove_node)
            df = df[df['link_id'] != remove_node]
        print(f'Remove nodes {remove_list}')

    def _optimal_gmm(self, x, n_components=None):
        if self.fitting_mode == 'iterate':
            lowest_bic = np.inf
            best_gmm = None
            if n_components is None:
                n_components = self.n_components

            for n in range(1, n_components + 1):
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

        elif self.fitting_mode == 'one-off':
            if n_components is None:
                n_components = self.n_components
            gmm = GaussianMixture(
                n_components=n_components,
                # covariance_type='diag',
                # init_params='k-means++',
                reg_covar=.2,
                n_init=3,
                random_state=66
            )
            gmm.fit(x)
            return gmm

    def fit_marginal_from_data(self):
        segment_gmms = {}
        for segment, data in self.speed_data.copy().groupby("link_id"):
            speeds = data["speed"].values.reshape(-1, 1)
            best_gmm = self._optimal_gmm(speeds,)
            segment_gmms[segment] = best_gmm
        self.marginals = segment_gmms
        return

    def fit_marginal_from_joints(self, upward=False):
        """
        Get marginal from the joint distributions. If the node has no parent nodes, get marginal
        from the joint where it is a parent node; if the node has parent nodes, get marginal
        from the joint where it is a child node.
        """
        all_nodes = set(self.joints.keys())  # get all nodes mentioned
        for entry in self.joints.values():
            all_nodes.update(entry['parents'])

        marginals = {}

        for node in all_nodes:
            found = False

            if not upward:
                if node in self.joints:
                    case = 1
                else:
                    case = 2
            else:
                if node not in set([i for k, v in self.joints.items() for i in v['parents']]):
                    case = 1
                else:
                    case = 2

            if case == 1:  # Case 1: marginalize from {node: joint}
                gmm = self.joints[node]['joints']
                marginals[node] = marginalize_gmm(gmm, 0)
                found = True
            elif case == 2:  # Case 2: marginalize from {the child of the node: joint}
                for child, info in self.joints.items():
                    if node in info['parents']:
                        gmm = info['joints']
                        var_order = [child] + info['parents']
                        idx = var_order.index(node)
                        marginals[node] = marginalize_gmm(gmm, idx)
                        found = True
                        break
            assert found, f"Could not find a joint distribution for node {node}"

        if not upward:
            self.marginals_downward = marginals
        else:
            self.marginals_upward = marginals
        return

    def fit_joint(self):
        df = self.speed_data.copy()
        dependency_gmms = {}
        for child in self.network.nodes():
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

            if len(parent_data) == 0 or len(child_data) == 0:
                continue

            x = child_data.values.reshape(-1, 1)
            y = parent_data.values  # keep the child at the first dim

            gmm = GaussianMixture(
                n_components=self.n_components,
                random_state=66,
                reg_covar=.2,
                n_init=3,
            )
            gmm.fit(np.hstack((x, y)))
            dependency_gmms[child] = {
                'parents': parents_filtered, 'joints': gmm, 'parent_type': self.network_mode
            }
        self.joints = dependency_gmms
        return

    def fit_signal(
            self, segment_flood_time, flood_prob,
            mode='from_marginal', verbose=0, upward=False, signal_filter=None
    ):
        import warnings

        def get_gmm(df):
            if (len(df)) > 10 and (df.sum() != 0):
                gmm = self._optimal_gmm(df.values.reshape(-1, 1), )
            else:
                gmm = None
                if verbose > 0:
                    warnings.warn(f'Sample size is too small. Skipped.')
            return gmm

        assert mode in ['from_marginal', 'from_data']
        speed = self.speed_data.copy()
        dists = {}
        for _, row in flood_prob.iterrows():
            link_id = row['link_id']
            p_flood = row['p']

            # Subset speed data for this link; if missing, record empty result and continue
            speed_data = speed[speed['link_id'] == link_id].copy()
            if speed_data.empty or link_id not in segment_flood_time:
                # dists[link_id] = {
                #     'p_flood': p_flood,
                #     'speed_no_flood': None,
                #     'speed_flood': None
                # }
                continue

            # Mark rows that fall within any flood event buffer window
            flood_events = segment_flood_time[link_id]
            speed_data['flooded'] = False
            event_wise_dist = []
            speed_data_original = speed_data.copy()

            # For each recorded flood event, label observations during the buffered window
            for _, event in flood_events.iterrows():
                mask = (speed_data['time'] >= event['buffer_start']) & (speed_data['time'] <= event['buffer_end'])
                speed_data.loc[mask, 'flooded'] = True

                # If signal_filter is requested, collect per-event GMMs to measure event-to-event consistency
                if signal_filter is not None:
                    speed_data_original.loc[mask, 'flooded'] = True
                    d = speed_data_original[speed_data_original['flooded']]['speed']
                    dist = get_gmm(d)
                    event_wise_dist.append(dist)

            # If a consistency filter is provided, compare pairwise KL between event GMMs
            if signal_filter is not None:
                from itertools import combinations
                dist_list = []

                # If only one event, skip (not enough to compare)
                if len(event_wise_dist) == 1:
                    # dist_list.append(0)
                    # dists[link_id] = {
                    #     'p_flood': p_flood,
                    #     'speed_no_flood': None,
                    #     'speed_flood': None
                    # }
                    continue

                # Compute KL divergences between each pair of event GMMs; missing GMMs contribute 0
                for gmm_a, gmm_b in combinations(event_wise_dist, 2):
                    if gmm_a is None or gmm_b is None:
                        dist_list.append(0)
                    else:
                        dist_list.append(calculate_kl_divergence_gmm(gmm_a, gmm_b))

                # Average divergence; if too large, treat link as unreliable and skip
                ave_dist = sum(dist_list) / len(dist_list)
                if ave_dist > signal_filter:
                    # dists[link_id] = {
                    #     'p_flood': p_flood,
                    #     'speed_no_flood': None,
                    #     'speed_flood': None
                    # }
                    continue

            # Prepare flood / non-flood samples depending on selected mode
            speed_no_flood, speed_flood = None, None
            if mode == 'from_data':
                # Directly use observed speeds labeled as flooded / not flooded
                speed_flood = speed_data[speed_data['flooded']]['speed']
                speed_no_flood = speed_data[~speed_data['flooded']]['speed']

            elif mode == 'from_marginal':

                # Sample synthetic observations from fitted marginals, then split by the flooded labels
                if upward:
                    assert self.marginals_upward is not None, 'Marginals have not been fit yet'
                    marginal = self.marginals_upward[link_id]
                else:
                    assert self.marginals_downward is not None, 'Marginals have not been fit yet'
                    marginal = self.marginals_downward[link_id]
                samples = marginal.sample(len(speed_data))[0]

                speed_data = speed_data.sort_values(by='speed')
                samples = np.sort(samples, axis=0)
                speed_flood = pd.Series(samples[speed_data['flooded'].to_numpy()][:, 0])
                speed_no_flood = pd.Series(samples[~speed_data['flooded'].to_numpy()][:, 0])

                if speed_flood.empty or speed_no_flood.empty:
                    continue
            
            # Fit GMMs for flooded and non-flooded samples (may return None if insufficient data)
            gmm_no_flood = get_gmm(speed_no_flood)
            gmm_flood = get_gmm(speed_flood)

            # Store results for this link
            dists[link_id] = {
                'p_flood': p_flood,
                'speed_no_flood': gmm_no_flood,
                'speed_flood': gmm_flood
            }
        if not upward:
            self.signal_downward = dists
        if upward:
            self.signal_upward = dists
        return

    def update_joints_legacy(
            self, orig_joint, orig_xk, observed_xk, xk_name, all_parent_ids,
    ):
        # sample from original joint, p(A, B_1, B_2), for example
        joint_samples = orig_joint.sample(self.n_samples)[0]
        xk_idx = all_parent_ids.index(xk_name) + 1  # +1 because child is first
        xk_samples = joint_samples[:, [xk_idx]]

        # get the ratio term, p'(B_1)/p(B_1)
        log_weights = observed_xk.score_samples(xk_samples) - orig_xk.score_samples(xk_samples)
        weights = np.exp(log_weights - np.max(log_weights))
        weights /= weights.sum()

        # apply the ratio term
        indices = np.random.choice(self.n_samples, size=self.n_samples, p=weights, replace=True)
        return joint_samples[indices]


    @staticmethod
    def gmm_pdf(gmm, y_grid):
        # part of the quick algo
        from scipy.stats import norm
        pdf = np.zeros_like(y_grid)
        for weight, mean, var in zip(gmm.weights_, gmm.means_.flatten(), gmm.covariances_.flatten()):
            pdf += weight * norm.pdf(y_grid, loc=mean, scale=np.sqrt(var))
        return pdf

    def update_joints_multi_variable_backup(
            self, orig_joint, orig_xk_list: list, observed_xk_list: list, xk_name_list: list, all_ids: list,
    ):
        # quick algo, not fully integrated
        orig_xk = orig_xk_list[0]
        observed_xk = observed_xk_list[0]
        obs_var_idx = 1
        # Prepare new weights
        new_weights = []

        for k in range(orig_joint.n_components):
            # Extract mean and variance of Y-dim in this component
            mean_Y = orig_joint.means_[k, obs_var_idx]
            var_Y = orig_joint.covariances_[k][obs_var_idx, obs_var_idx]
            std_Y = np.sqrt(var_Y)

            # Evaluate correction factor at mean_Y
            pY = self.gmm_pdf(orig_xk, np.array([mean_Y]))[0]
            tilde_pY = self.gmm_pdf(observed_xk, np.array([mean_Y]))[0]

            correction_factor = tilde_pY / (pY + 1e-12)  # avoid division by zero

            # New weight = old weight * correction
            new_weight = orig_joint.weights_[k] * correction_factor
            new_weights.append(new_weight)

        # Normalize weights
        new_weights = np.array(new_weights)
        new_weights /= np.sum(new_weights)

        # Build new GMM with same means and covariances but updated weights
        new_joint = GaussianMixture(
            n_components=orig_joint.n_components,
            covariance_type='full'
        )

        # Manually set parameters
        new_joint.weights_ = new_weights
        new_joint.means_ = np.copy(orig_joint.means_)
        new_joint.covariances_ = np.copy(orig_joint.covariances_)

        # The sklearn GMM also needs precisions_cholesky_
        new_joint.precisions_cholesky_ = orig_joint.precisions_cholesky_
        return new_joint

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

    def update_network_with_soft_evidence_legacy(
            self, link_name, observed_marginal, marginals, joints, verbose=1
    ):
        """
        [IMPORTANT NOTE]
        For nodes with multiple parent nodes, the algo below update the corresponding joint distribution
        at separate steps depending on which direction does the information arrives first.
        It would derive a plausible result in the dataset under development, but it is NOT fully correct.
        """
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
                resampled_joint_samples = self.update_joints_legacy(
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

    def update_joints_multi_variable(
            self, orig_joint, orig_xk_list: list, observed_xk_list: list, xk_name_list: list, all_ids: list,
            avoid_numerical_issue=True
    ):
        """
        xk is the parent node who gets new information
        orig_xk_list: original marginals of xk
        observed_xk_list: observed marginals of xk
        xk_name: the names of xk
        all_ids: the names of all parent nodes, may be a big set than xk
        """

        assert len(orig_xk_list) == len(observed_xk_list) == len(xk_name_list)

        # sample from original joint, p(A, B_1, B_2), for example
        joint_samples = orig_joint.sample(self.n_samples)[0]
        xk_idx_list = [all_ids.index(name) for name in xk_name_list]
        xk_samples = joint_samples[:, xk_idx_list]

        # get the ratio term, p'(B_1)/p(B_1) or p'(B_1 B_2)/p(B_1 B_2,) etc.
        # in the Bayesian Network, B_1 and B_2 are assumed to be independent
        observed_xk_score = 0
        for observed_xk, i in zip(observed_xk_list, range(xk_samples.shape[1])):
            observed_xk_score += observed_xk.score_samples(xk_samples[:, [i]])

        orig_xk_score = 0
        for orig_xk, i in zip(orig_xk_list, range(xk_samples.shape[1])):
            orig_xk_score += orig_xk.score_samples(xk_samples[:, [i]])

        sample_size = self.n_samples
        if avoid_numerical_issue:
            # remove the bottom 1% samples (who could have very low probability) to avoid numerical issue
            bottom_1_percent_indices = np.where(orig_xk_score <= np.percentile(orig_xk_score, 1))[0]
            joint_samples = np.delete(joint_samples, bottom_1_percent_indices, axis=0)
            observed_xk_score = np.delete(observed_xk_score, bottom_1_percent_indices, axis=0)
            orig_xk_score = np.delete(orig_xk_score, bottom_1_percent_indices, axis=0)
            sample_size = self.n_samples - bottom_1_percent_indices.size

        log_weights = observed_xk_score - orig_xk_score
        weights = np.exp(log_weights - np.max(log_weights))
        weights /= weights.sum()

        # apply the ratio term
        indices = np.random.choice(sample_size, size=sample_size, p=weights, replace=True)
        return joint_samples[indices]

    def update_network_with_multiple_soft_evidence_downward(
            self, signal_dict: dict, marginals, joints, verbose=1,
    ):
        import copy
        from collections import defaultdict, deque

        marginals_updated = copy.deepcopy(marginals)
        marginals_fixed = copy.deepcopy(marginals)
        joints_fixed = copy.deepcopy(joints)
        joints_updated = copy.deepcopy(joints)
        updated_marginals = []

        signal_dict = {k: v for k, v in signal_dict.items() if v is not None}
        if not signal_dict:
            raise ValueError('No signal.')

        # traverse the graph and record the parents affected by information propagation for each node
        reachable = defaultdict(list)  # records reachable parents by the information propagation
        queue_0 = deque(set(signal_dict.keys()))
        visited_0 = set()
        while queue_0:
            current_node_0 = queue_0.popleft()
            if current_node_0 in visited_0:
                continue
            visited_0.add(current_node_0)
            for child_0 in self.network.successors(current_node_0):
                reachable[child_0].append(current_node_0)
                queue_0.append(child_0)

        remaining_parents_n = {
            node: len([p for p in self.network.predecessors(node) if p in reachable[node]])
            for node in self.network.nodes
        }

        # core algo below
        queue = deque(set(signal_dict.keys()))
        visited = set()
        for n, gmm in signal_dict.items():
            marginals_updated[n] = gmm
            updated_marginals.append(n)

        while queue:
            current_node = queue.popleft()
            if current_node in visited:
                continue
            visited.add(current_node)

            for child in self.network.successors(current_node):
                assert child in joints_fixed

                remaining_parents_n[child] -= 1
                if remaining_parents_n[child] > 0:
                    continue  # not all reachable parents are ready

                # update joints and marginals
                parent_ids, joint_gmm = joints_fixed[child]['parents'], joints_fixed[child]['joints']
                assert current_node in parent_ids
                resampled_joint_samples = self.update_joints_multi_variable(
                    joint_gmm,
                    [marginals_fixed[p] for p in reachable[child]],
                    [marginals_updated[p] for p in reachable[child]],
                    reachable[child], [child] + parent_ids,
                )

                resampled_child_samples = resampled_joint_samples[:, [0]]
                new_child_gmm = self._optimal_gmm(resampled_child_samples,)
                marginals_updated[child] = new_child_gmm
                updated_marginals.append(child)

                new_joint_gmm = self._optimal_gmm(resampled_joint_samples)
                joints_updated[child]['joints'] = new_joint_gmm

                if verbose > 0:
                    print(f"Updated marginal for {child}")

                queue.append(child)

        return marginals_updated, updated_marginals, joints_updated

    def update_network_with_multiple_soft_evidence_upward(
            self, signal_dict: dict, marginals, joints, verbose=1,
    ):
        import copy
        from collections import defaultdict, deque

        marginals_updated = copy.deepcopy(marginals)
        marginals_fixed = copy.deepcopy(marginals)
        joints_fixed = copy.deepcopy(joints)
        joints_updated = copy.deepcopy(joints)
        updated_marginals = []

        signal_dict = {k: v for k, v in signal_dict.items() if v is not None}
        if not signal_dict:
            raise ValueError('No signal.')

        # traverse the graph and record the parents affected by information propagation for each node
        reachable = defaultdict(list)  # records reachable parents by the information propagation
        queue_0 = deque(set(signal_dict.keys()))
        visited_0 = set()
        while queue_0:
            current_node_0 = queue_0.popleft()
            if current_node_0 in visited_0:
                continue
            visited_0.add(current_node_0)
            for child_0 in self.network.predecessors(current_node_0):
                reachable[child_0].append(current_node_0)
                queue_0.append(child_0)

        remaining_parents_n = {
            node: len([p for p in self.network.successors(node) if p in reachable[node]])
            for node in self.network.nodes
        }

        # core algo below
        queue = deque(set(signal_dict.keys()))
        visited = set()
        for n, gmm in signal_dict.items():
            marginals_updated[n] = gmm

        while queue:
            current_node = queue.popleft()
            if current_node in visited:
                continue
            visited.add(current_node)

            for child in self.network.predecessors(current_node):
                if child not in joints_fixed:
                    continue  # in upward update, the top node has no parents

                remaining_parents_n[child] -= 1
                if remaining_parents_n[child] > 0:
                    continue  # not all reachable parents are ready

                # update joint
                parent_ids = joints_fixed[current_node]['parents']
                joint_gmm = joints_fixed[current_node]['joints']

                resampled_joint_samples = self.update_joints_multi_variable(
                    joint_gmm,
                    [marginals_fixed[current_node]],
                    [marginals_updated[current_node]],
                    [current_node], [current_node] + parent_ids,
                )

                new_joint_gmm = self._optimal_gmm(resampled_joint_samples)
                joints_updated[current_node]['joints'] = new_joint_gmm

                # update marginal
                for p in parent_ids:
                    all_id = [current_node] + parent_ids
                    idx = all_id.index(p)
                    resampled_child_samples = resampled_joint_samples[:, [idx]]
                    new_child_gmm = self._optimal_gmm(resampled_child_samples)
                    marginals_updated[child] = new_child_gmm
                    updated_marginals.append(child)

                if verbose > 0:
                    print(f"Updated marginal for {child}")

                queue.append(child)

        return marginals_updated, updated_marginals, joints_updated

    def update_network_with_multiple_soft_evidence(
            self, signals: list, marginals, joints, verbose=1,
    ):
        # OUTPUTS:
        # marginals: combined marginals on two directions. A loc only has one marginal
        # joints: updated joints that could be used as the next BN
        # marginals_up_down: updated marginals with two directions, that can be used for the next BN

        signal_down = signals[0]
        signal_up = signals[1]

        marginal_down = marginals[0]
        marginal_up = marginals[1]

        marginals_down, updated_loc_down, joints_updated_down = self.update_network_with_multiple_soft_evidence_downward(
            signal_down, marginal_down, joints, verbose=verbose
        )
        marginals_up, updated_loc_up, joints_updated_up = self.update_network_with_multiple_soft_evidence_upward(
            signal_up, marginal_up, joints, verbose=verbose,
        )

        # combine updated locs
        locs_updated = {'down': updated_loc_down, 'up': updated_loc_up}

        # combine the updated marginals on two directions
        marginals = marginals_down.copy()
        for l in updated_loc_up:
            marginals[l] = marginals_up[l]

        # combine the updated joints on two directions
        joints = joints_updated_down.copy()
        for l in updated_loc_up:
            joints[l] = joints_updated_up[l]

        # update marginal_up where marginal_down is updated, and vice versa
        # the marginals at both direction have been updated at the signal locs
        marginals_up_down = {'down': marginals_down, 'up': marginals_up}
        marginals_up_down = update_marginal_on_op_direction(
            marginals_up_down, 
            {'down': [i for i in updated_loc_down if i not in list(signal_down.keys())], 
             'up': [i for i in updated_loc_up if i not in list(signal_down.keys())]}, 
            joints
        )

        return marginals, locs_updated, joints, marginals_up_down

    def calculate_gmm_entropy_approximation(self, gmm):
        samples, _ = gmm.sample(self.n_samples)
        log_probs = gmm.score_samples(samples)
        entropy = -np.mean(log_probs)
        return entropy

    def calculate_network_entropy(self, marginals=None, joints=None, verbose=1):
        # if the node has no parent, the marginal was used
        # if the node has parents, joint and the marginal refitted from joint sampling was used

        if marginals is None:
            marginals = self.marginals_downward
        if joints is None:
            joints = self.joints

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

    def calculate_network_conditional_entropy(self, network_list, verbose=1, label=None):
        """
        Conditional entropy: H(A∣B)=∑_b P(B=b)H(A∣B=b)
        Example: 0.01 * entropy {network updated with flood time dist_w_obs}
        + 0.99 * entropy {network with non-flood time dist_w_obs}

        Example command of this method
        entropies[k] = bayes_network_t.calculate_network_conditional_entropy([
            {'p': 1 - v['p_flood'], 'marginals': marginals_no_flood, 'joints': joints_no_flood},
            {'p': v['p_flood'], 'marginals': marginals_flood, 'joints': joints_flood},
        ], label=k)
        or simply use the original network stored in the class
        entropy_original = bayes_network_t.calculate_network_entropy()
        """
        entropy = 0
        for n in network_list:
            entropy += n['p'] * self.calculate_network_entropy(
                marginals=n['marginals'], joints=n['joints'], verbose=0,
            )
        if verbose > 0:
            if label is not None:
                print(f"Total Entropy of the Bayesian Network w. observation {label}: {entropy}")
            else:
                print(f"Total Entropy of the Bayesian Network: {entropy}")
        return entropy

    def calculate_network_conditional_kl_divergence(self, network_list, update_loc, verbose=1, label=None):
        assert len(network_list), 'Divergence is between two networks'
        network_0, network_1 = network_list[0], network_list[1]
        update_loc_0, update_loc_1 = update_loc[0], update_loc[1]

        divergence_0 = 0
        integrated_marginal_0 = self.marginals_downward.copy()
        for l in update_loc_0['up']:  # loc updated by upward update
            integrated_marginal_0[l] = self.marginals_upward[l]
        for prior, posterior in zip(list(integrated_marginal_0.values()), list(network_0['marginals'].values())):
            divergence_0 += calculate_kl_divergence_gmm(posterior, prior)

        divergence_1 = 0
        integrated_marginal_1 = self.marginals_downward.copy()
        for l in update_loc_1['up']:  # loc updated by upward update
            integrated_marginal_1[l] = self.marginals_upward[l]
        for prior, posterior in zip(list(integrated_marginal_1.values()), list(network_1['marginals'].values())):
            divergence_1 += calculate_kl_divergence_gmm(posterior, prior)

        divergence = divergence_0 * network_0['p'] + divergence_1 * network_1['p']
        if verbose > 0:
            if label is not None:
                print(f"KL divergencey of the two Bayesian Network w. observation {label}: {divergence}")
            else:
                print(f"KL divergencey of the two Bayesian Network: {divergence}")
        return divergence

    @staticmethod
    def calculate_network_kl_divergence(network_list, verbose=1, label=None, expand=False):
        assert len(network_list), 'Divergence is between two networks'
        network_0, network_1 = network_list[0], network_list[1]  # estimate, ground truth
        common_keys = network_0.keys() & network_1.keys()

        divergence_list = []
        computed = False
        for k in common_keys:
            divergence_list.append(calculate_kl_divergence_gmm(network_0[k], network_1[k]))
            computed = True
        divergence = sum(divergence_list)
        if divergence < 1e-8:
            divergence = 0.0

        if not computed:
            warnings.warn('No KL divergence is computed.')
            return None

        if verbose > 0:
            if label is not None:
                print(f"KL divergencey of the two Bayesian Network w. observation {label}: {divergence}")
            else:
                print(f"KL divergencey of the two Bayesian Network: {divergence}")
        if expand == False:
            return divergence
        if expand == True:
            return divergence_list

    @staticmethod
    def calculate_multi_network_kl_divergence(network_list, verbose=1, label=None):
        assert len(network_list), 'Divergence is between two networks'
        network_0, network_1 = network_list[0], network_list[1]  # estimate, ground truth
        common_keys = network_0.keys() & network_1.keys()

        divergence_list = []
        computed = False
        for k in common_keys:
            divergence_list.append(calculate_kl_divergence_gmm(network_0[k], network_1[k]))
            computed = True
        divergence = sum(divergence_list)

        if not computed:
            warnings.warn('No KL divergence is computed.')
            return None

        if verbose > 0:
            if label is not None:
                print(f"KL divergencey of the two Bayesian Network w. observation {label}: {divergence}")
            else:
                print(f"KL divergencey of the two Bayesian Network: {divergence}")

        return divergence

    def get_entropy_with_signals(self, bayes_joints, bayes_marginal, signal_dict, if_vis=False):
        entropies = {}
        for k, v in signal_dict.items():
            if (v['speed_no_flood'] is not None) and (v['speed_flood'] is not None):
                marginals_no_flood, joints_no_flood = self.update_network_with_soft_evidence_legacy(
                    k, v['speed_no_flood'], bayes_marginal, bayes_joints, verbose=0
                )
                marginals_flood, joints_flood = self.update_network_with_soft_evidence_legacy(
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

    def convert_state_to_dist(self, inferred_states):
        inferred_states_filtered = {k: v for k, v in inferred_states.items() if k in list(self.signal_downward.keys())}
        signal_down = {
            k: (
                self.signal_downward[k]['speed_flood']
                if v == 'flooded'
                else self.signal_downward[k]['speed_no_flood']
                if v == 'not flooded'
                else None
            )
            for k, v in inferred_states_filtered.items()
        }
        signal_up = {
            k: (
                self.signal_downward[k]['speed_flood']
                if v == 'flooded'
                else self.signal_downward[k]['speed_no_flood']
                if v == 'not flooded'
                else None
            )
            for k, v in inferred_states_filtered.items()
        }
        return [signal_down, signal_up]

    def save_instance(self, file):
        import pickle
        from pathlib import Path
        file_path = Path(file)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        file = Path(file)
        with file.open("wb") as f:
            pickle.dump(self, f)

    def load_instance(self, file):
        try:
            import pickle
            from pathlib import Path
            file = Path(file)
            with file.open("rb") as f:
                loaded = pickle.load(f)
            self.__dict__.clear()
            self.__dict__.update(loaded.__dict__)
            print('Traffic Bayesian Network exists')
            return True
        except Exception as e:
            print('Failed to load Traffic Bayesian Network, build from scratch')
            return False

    def check_keys(self):
        # check consistency between joints and the graph
        assert set(self.joints.keys()).issubset(self.network.nodes())  # joints are a subset of graph
        mentioned_nodes = set(self.joints.keys())  # dict keys
        for value in self.joints.values():
            mentioned_nodes.update(value.get('parents', []))  # add parents
        assert mentioned_nodes == set(self.network.nodes), "All nodes mentioned by joints must equal all graph nodes"

        # all nodes in the graph should have marginals
        common_keys = set(self.network.nodes()) & set(self.marginals_downward.keys())
        self.marginals_downward = {k: v for k, v in self.marginals_downward.items() if k in common_keys}

        common_keys = set(self.network.nodes()) & set(self.marginals_upward.keys())
        self.marginals_upward = {k: v for k, v in self.marginals_upward.items() if k in common_keys}

        # filter out signals without both flood and non-flood GMMs fitted
        self.signal_downward = {
            k: v for k, v in self.signal_downward.items()
            if v['speed_flood'] is not None #and v['speed_no_flood'] is not None
        }
        common_keys = set(self.network.nodes()) & set(self.signal_downward.keys())
        self.signal_downward = {k: v for k, v in self.signal_downward.items() if k in common_keys}

        self.signal_upward = {
            k: v for k, v in self.signal_upward.items()
            if v['speed_flood'] is not None #and v['speed_no_flood'] is not None
        }
        common_keys = set(self.network.nodes()) & set(self.signal_upward.keys())
        self.signal_upward = {k: v for k, v in self.signal_upward.items() if k in common_keys}

class FloodBayesNetwork:
    def __init__(self, t_window: str = 'D'):
        """
        :param t_window: the unit for processing flood data, default is day
        """
        self.t_window = t_window  # time window to locate a flooding
        self.network = None
        self.marginals = None
        self.conditionals = None
        self.network_bayes = None

    def fit_marginal(self, df: pd.DataFrame, if_vis=False):
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

    def build_network_by_co_occurrence(
            self, df: pd.DataFrame, weight_thr: float = 0, edge_thr: int = 1, report: bool = False
    ):
        """
        df should contain time_create and link_id col
        time_create col is the start time of the road closure
        link_id col is the id of the road

        weight_thr: edges with weights below the threshold will be removed

        edge_thr: build an edge if the number of co-occurring floods exceeds the threshold

        Note that the network topology is not perfectly defined. Please consider the facts below:

        Co-occurrence network is used to define the topology, and it is problematic when the
        occurrence of flooding is rare because of the lack of samples.

        The algo below would produce double connections between two nodes, e.g., A to B and B to A,
        so the connection with lower weight was removed.

        The temporal dimension is not considered. B happens after A (not just co-occurrence) has more
        useful information.

        Considering the elevation of roads could be helpful.
        """

        _, occurrence, co_occurrence = self.process_raw_flood_data(df.copy())

        graph = nx.DiGraph()
        graph.add_nodes_from(df.copy()['link_id'].unique())
        for node in graph.nodes:
            graph.nodes[node]['occurrence'] = occurrence[node]
        for (a, b), count in co_occurrence.items():
            if count >= edge_thr:
                prob = count / occurrence[a]  # if occurrence / co-occurrence is over the thr, there is an edge
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

    def fit_conditional(self, df: pd.DataFrame):
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

    def infer_w_evidence(self, target_node: str, evidence: dict):
        """
        Get the probability of flooding for roads.
        Example input:
            target_node = 'A'
            evidence = {'B': 1, 'C': 0}, where 1 = flooded, 0 = not flooded
        """
        from pgmpy.inference import VariableElimination
        assert target_node in self.network_bayes.nodes, f"{target_node} is not in the network."
        assert isinstance(evidence, dict), "evidence must be a dictionary."
        for node, state in evidence.items():
            assert isinstance(node, str), f"Evidence key '{node}' must be a string."
            assert node in self.network_bayes.nodes, f"Evidence node '{node}' is not in the network."
            assert state in [0, 1], f"Evidence value for '{node}' must be 0 or 1, got {state}."

        inference = VariableElimination(self.network_bayes)
        result = inference.query(variables=[target_node], evidence=evidence)
        p = result.values
        return {'not_flooded': p[0], 'flooded': p[1]}

    def infer_node_states(self, node, node_value, thr_flood, thr_not_flood):
        """
        Given the observation of a node, get other nodes with a flooding probability above the threshold.
        """
        flooded_nodes = []
        not_flooded_nodes = []
        if node in self.network_bayes.nodes():
            for n in self.network_bayes.nodes():

                if self.network_bayes.in_degree(n) == 0:
                    continue
                if n == node:
                    continue

                p = self.infer_w_evidence(n, {node: node_value})
                if p['flooded'] >= thr_flood:
                    flooded_nodes.append(n)
                if p['not_flooded'] >= thr_not_flood:
                    not_flooded_nodes.append(n)

        assert not set(flooded_nodes) & set(not_flooded_nodes), """
        At least one road is regarded both flooded and not flooded
        """
        output = {i: 'flooded' for i in flooded_nodes}
        output.update({i: 'not flooded' for i in not_flooded_nodes})
        return output

    @staticmethod
    def remove_min_weight_feedback_arcs(graph: nx.DiGraph()):
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

    def process_raw_flood_data(self, df: pd.DataFrame):
        """
        Calculate the flood incidents time bin, the occurrence of the flooding on each segment,
        and the co-occurrence count
        :param df: flood induced road closure datasets
        :return: df, the flooded roads in each incident
                dict, the flood count for roads
                dict the co-occurrence count for each pair
        """
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

    def save_instance(self, file):
        import pickle
        from pathlib import Path
        file_path = Path(file)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        file = Path(file)
        with file.open("wb") as f:
            pickle.dump(self, f)

    def load_instance(self, file):
        try:
            import pickle
            from pathlib import Path
            import joblib
            file = Path(file)
            loaded = joblib.load(file)
            # with file.open("rb") as f:
            #     loaded = pickle.load(f)
            self.__dict__.clear()
            self.__dict__.update(loaded.__dict__)
            print('Flood Bayesian Network exists')
            return True
        except Exception as e:
            print('Failed to load Flood Bayesian Network, build from scratch')
            return False


class MultiGaussianMixture:
    """A simple wrapper to combine multiple GMMs with probabilities"""
    
    def __init__(self, gmms_with_probs):
        """
        Args:
            gmms_with_probs: List of (GaussianMixture, probability) tuples
        """
        self.gmms, self.probs = zip(*gmms_with_probs)
        self.probs = np.array(self.probs)
        
        # Validate probabilities sum to 1
        # This could be violated due to the filtering out low probability cases
        assert np.isclose(self.probs.sum(), 1.0, rtol=1e-3, atol=1e-3), f"Probabilities must sum to 1, got {self.probs.sum()}"
    
    def sample(self, n_samples):
        """Sample from the mixture of GMMs"""
        # Determine how many samples from each GMM based on probabilities
        n_samples_per_gmm = np.random.multinomial(n_samples, self.probs)
        
        # Sample from each GMM
        samples = []
        for gmm, n in zip(self.gmms, n_samples_per_gmm):
            if n > 0:
                X, _ = gmm.sample(n)
                samples.append(X)
        
        # Combine all samples
        X_combined = np.vstack(samples)
        
        # Shuffle to mix samples from different GMMs
        indices = np.random.permutation(len(X_combined))
        return X_combined[indices], None
    
    def score_samples(self, X):
        from scipy.special import logsumexp
        """Calculate log probability of samples under the mixture"""
        # Calculate log prob under each GMM: shape (n_gmms, n_samples)
        log_probs = np.array([gmm.score_samples(X) for gmm in self.gmms])
        
        # Mixture log probability: log(sum(p_i * P_i(x)))
        # = logsumexp(log(p_i) + log(P_i(x)))
        log_mixture_prob = logsumexp(
            log_probs + np.log(self.probs)[:, np.newaxis], 
            axis=0
        )
        
        return log_mixture_prob


def calculate_kl_divergence_gmm(P_gmm, Q_gmm, n_samples=100000):
    X_samples, _ = P_gmm.sample(n_samples)
    log_p = P_gmm.score_samples(X_samples)
    log_q = Q_gmm.score_samples(X_samples)
    kl_div = np.mean(log_p - log_q)
    return kl_div

def marginalize_gmm(gmm, var_idx):
    # part of the quick algo
    n_components = gmm.n_components

    # Build new 1D GMM manually
    marginal_gmm = GaussianMixture(n_components=n_components, covariance_type='full')

    # Set parameters
    marginal_gmm.weights_ = np.copy(gmm.weights_)

    # Extract 1D means and variances
    marginal_means = gmm.means_[:, var_idx].reshape(-1, 1)
    marginal_covariances = np.zeros((n_components, 1, 1))

    for k in range(n_components):
        var = gmm.covariances_[k][var_idx, var_idx]
        marginal_covariances[k, 0, 0] = var

    marginal_gmm.means_ = marginal_means
    marginal_gmm.covariances_ = marginal_covariances
    marginal_gmm.precisions_cholesky_ = 1.0 / np.sqrt(marginal_covariances)

    return marginal_gmm


def check_gmr_bn_consistency(node_list: list, joints: dict):
    for node in node_list:
        if node in ['4616305', '4616323', '4616351', '4616353', '4620314', '4620330', '4620331', '4620332']:
            print(f'Check node {node}')
            joints_list = []
            for k, v in joints.items():
                if node == k or (node in v['parents']):
                    joints_list.append({k: v})
            if len(joints_list) > 1:
                for j in joints_list:
                    joint = list(j.values())[0]['joints']
                    v_list = list(j.keys()) + list(j.values())[0]['parents']
                    node_index = v_list.index(node)
                    marginal = marginalize_gmm(joint, node_index)
                    vis.dist_gmm_1d(marginal, title=node)


def norm_n_weight(results, weight_disruption, pre_defined_norm_bounds=None):     
    # get norm bounds if pre-defined
    if pre_defined_norm_bounds is not None:
        d_min, d_max, uod_min, uod_max = pre_defined_norm_bounds

    # Filter out None values and track valid indices
    expand_results = [v for _, v in results.items() if v['info_gain_disruption_weighted'] is not None]

    # Calculate normalized values
    v_dirp = [r['info_gain_disruption_weighted'] for r in expand_results]
    if pre_defined_norm_bounds is None:
        v_dirp_norm = {k['road']: (v - min(v_dirp)) / (max(v_dirp) - min(v_dirp)) if max(v_dirp) != min(v_dirp) else 0 for v, k in zip(v_dirp, expand_results)}
    else:
        v_dirp_norm = {k['road']: (v - d_min) / (d_max - d_min) if d_max != d_min else 0 for v, k in zip(v_dirp, expand_results)}
    
    v_sup = [r['info_gain_suprise'] for r in expand_results]
    if pre_defined_norm_bounds is None:
        v_sup_norm = {k['road']: (v - min(v_sup)) / (max(v_sup) - min(v_sup)) if max(v_sup) != min(v_sup) else 0 for v, k in zip(v_sup, expand_results)}
    else:
        v_sup_norm = {k['road']: (v - uod_min) / (uod_max - uod_min) if uod_max != uod_min else 0 for v, k in zip(v_sup, expand_results)}

    # Calculate objective and find max
    v_obj = {k['road']: d * weight_disruption + s * (1 - weight_disruption) for d, s, k in zip(v_dirp_norm.values(), v_sup_norm.values(), expand_results)}
    max_v_obj = max(v_obj.values())
    selected_road = [k for k, v in v_obj.items() if v == max_v_obj]
    selected_road = selected_road[0]

    valid_results = {k: v for i, (k, v) in enumerate(results.items())
                if v['info_gain_disruption_weighted'] is not None}
    if not valid_results:
        raise ValueError("All disruption values are None")
    for k, v in v_dirp_norm.items():
        valid_results[k]['info_gain_disruption_weighted_normed'] = v
    for k, v in v_sup_norm.items():
        valid_results[k]['info_gain_suprise_normed'] = v
    for k, v in v_obj.items():
        valid_results[k]['voi'] = v

    return selected_road, valid_results


def get_signals(k, v, bayes_network_t, bayes_network_f, thr_flood, thr_not_flood):
    inferred_signals_flood = bayes_network_t.convert_state_to_dist(
        bayes_network_f.infer_node_states(k, 1, thr_flood, thr_not_flood)
    )
    signals = [
            {**{k: v['speed_flood']}, **inferred_signals_flood[0]},  # downward
            {**{k: bayes_network_t.signal_upward[k]['speed_flood']}, **inferred_signals_flood[1]},  # upward
        ]
    return signals

def update_marginal_on_op_direction(marginals_up_down, locs_updated, joints):
    """
    In the downward update, only downward marginals are updated. However, upward marginals at the same loc should 
    also be updated. The same case for the upward update.

    marginals_up_down: a dict of all marginals for all nodes (some nodes are not updated)
    locs_updated: a list of updated locs except for the signal locs (take off signal locs before the input)
    joints: the joints for update
    """
    marginals_down = marginals_up_down['down']
    marginals_up = marginals_up_down['up']

    nodes_updated_by_down = locs_updated['down']
    nodes_updated_by_up = locs_updated['up']

    for node in nodes_updated_by_down:
    # process the node updated by downward
    # the upward marginal of the node should be updated
    # the upward marginal should be marginalized from the joint labelled by itself
        gmm = joints[node]['joints']
        marginals_up[node] = marginalize_gmm(gmm, 0)

    for node in nodes_updated_by_up:
    # opposite to the comment above
        for child, info in joints.items():
            if node in info['parents']:
                gmm = info['joints']
                var_order = [child] + info['parents']
                idx = var_order.index(node)
                marginals_down[node] = marginalize_gmm(gmm, idx)
                break

    return [marginals_down, marginals_up]


def edit_marginals_only(marginals_only):
    # check and edit belief_network_4_compute
    # just for keeping main.py clean

    # check
    assert len(marginals_only) >= 1
    assert all(t[1] == marginals_only[0][1] for t in marginals_only), "updated locs inconsistent"
    
    # norm the p in belief_network_4_compute
    sum_p = sum([i[2] for i in marginals_only])
    marginals_only = [(i[0], i[1], i[2] / sum_p) for i in marginals_only]
    
    # remove uncovered locs
    covered_locs_k = [*marginals_only[0][1]['down'], *marginals_only[0][1]['up']]
    marginals_only = [
        ({k: v for k, v in t[0].items() if k in covered_locs_k}, t[2])
        for t in marginals_only
    ]

    # convert to a dict of MultiGaussianMixture
    dict_multi_gmm = convert_gmms_to_multi_gmms(marginals_only)

    return dict_multi_gmm


def edit_belief_networks(belief_networks, down_loc, up_loc):
    # format
    belief_networks_marginals_down = [(
        dict(filter(lambda item: item[0] in  down_loc, i[1][0].items())),
        i[2],
    ) for i in belief_networks]
    belief_networks_marginals_up = [(
        dict(filter(lambda item: item[0] in up_loc, i[1][1].items())),
        i[2],
    ) for i in belief_networks]
    
    belief_networks_marginals_combined = [(i[0] | j[0], i[1]) for i, j 
                                          in zip(belief_networks_marginals_down, belief_networks_marginals_up)]
    multi_belief_networks_marginals = convert_gmms_to_multi_gmms(belief_networks_marginals_combined)
    return multi_belief_networks_marginals


def convert_gmms_to_multi_gmms(marginals):
    # conver to a dict of [(GMM1, p_1), (GMM_2, p2), .... (GMM_n, p_n)]
    key_sets = [set(bn_dict.keys()) for bn_dict, _ in marginals]
    assert all(keys == key_sets[0] for keys in key_sets), "Keys are not consistent across all networks"
    dict_multi_gmm_raw = {}
    for road in key_sets[0]:
        dict_multi_gmm_raw[road] = [(bn_dict[road], prob) for bn_dict, prob in marginals]

    # convert to a dict of MultiGaussianMixture
    dict_multi_gmm = {
        k: MultiGaussianMixture(gmm_list) 
        for k, gmm_list in dict_multi_gmm_raw.items()
    }
    return dict_multi_gmm


def safe_subtract(a, b):
    if a is None or b is None:
        return None
    return a - b

