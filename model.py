import warnings

import numpy as np
import os
os.environ["OMP_NUM_THREADS"] = "1"
from sklearn.mixture import GaussianMixture
import visualization as vis


class TrafficBayesNetwork:
    def __init__(self):
        self.network = None
        self.gmm_per_segment = None
        self.closure_p_per_segment = None
        self.gmm_joint_road_road = None
        self.gmm_joint_flood_road = None
        return

    @staticmethod
    def _optimal_gmm(x, max_components):
        lowest_bic = np.inf
        best_gmm = None
        for n in range(1, max_components + 1):
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

    def fit_speed(self, df, max_components=2):
        segment_gmms = {}
        for segment, data in df.groupby("link_id"):
            speeds = data["speed"].values.reshape(-1, 1)
            best_gmm = self._optimal_gmm(speeds, max_components)
            segment_gmms[segment] = best_gmm
        self.gmm_per_segment = segment_gmms
        return

    def fit_flood(self, closures, interval='1D', if_vis=False):
        if if_vis:
            vis.bar_closure_count_over_time(closures.copy())
        pass
        closures = closures.copy().set_index('time_create')
        closure_count = closures.groupby('link_id').count()['id']
        daily_counts = closures.resample(interval).size().fillna(0)
        flooding_day_count = len(daily_counts[daily_counts > 0])
        closure_count_p = closure_count / flooding_day_count
        closure_count_p = closure_count_p.reset_index().rename(columns={'id': 'p'})
        self.closure_p_per_segment = closure_count_p
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
                    edges.append((row["link_id"], downstream_segment))
        self.network = BayesianNetwork(edges)
        return

    def remove_no_data_segment_from_network(self, graph):
        node_list_graph = list(graph.nodes())
        node_list_gmm = list(self.gmm_per_segment.keys())
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

    def build_network_from_geo(self, gdf, remove_no_data_segment=True):
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

    def fit_joint_speed_n_speed(self, df, max_components=3):
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

            x = parent_data.values
            y = child_data.values.reshape(-1, 1)

            gmm = GaussianMixture(
                n_components=max_components,
                random_state=66,
                reg_covar=.2,
                n_init=3,
            )
            gmm.fit(np.hstack((x, y)))
            dependency_gmms[child] = (parents_filtered, gmm)
        self.gmm_joint_road_road = dependency_gmms
        return

    def fit_joint_flood_n_speed(self, flood_time, speed, max_components=2):
        import warnings

        dists = {}
        for _, row in self.closure_p_per_segment.iterrows():
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
                if len(df) > max_components:
                    gmm = self._optimal_gmm(df.values.reshape(-1, 1), max_components)
                elif len(df) > 0:
                    gmm = self._optimal_gmm(df.values.reshape(-1, 1), 1)
                else:
                    gmm = GaussianMixture(n_components=1, random_state=42)
                    gmm.means_ = np.array([[0.0]])
                    gmm.covariances_ = np.array([[[.05]]])
                    gmm.weights_ = np.array([1.0])
                    gmm.precisions_cholesky_ = np.array([[[1.0 / np.sqrt(1e-10)]]])
                    gmm.converged_ = True
                    warnings.warn('No speed data during the df time range. Zero speed assumed.')
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

    @staticmethod
    def calculate_gmm_entropy_approximation(gmm, n_samples=10000):
        samples, _ = gmm.sample(n_samples)
        log_probs = gmm.score_samples(samples)
        entropy = -np.mean(log_probs)
        return entropy

    def calculate_network_entropy(self):
        import networkx as nx

        topo_order = list(nx.topological_sort(self.network))
        total_entropy = 0.0
        for node in topo_order:
            parents = list(self.network.predecessors(node))
            if not parents:
                gmm = self.gmm_per_segment[node]
                entropy = self.calculate_gmm_entropy_approximation(gmm)
                total_entropy += entropy
                print(f"Node {node} (root): H({node}) = {entropy:.4f}")

            else:
                if node in self.gmm_joint_road_road:
                    parent_ids, joint_gmm = self.gmm_joint_road_road[node]
                    h_joint = self.calculate_gmm_entropy_approximation(joint_gmm)
                    if len(parents) == 1:
                        h_parents = self.calculate_gmm_entropy_approximation(self.gmm_per_segment[parents[0]])
                    else:
                        # NOTE: Joint dist of parent nodes is needed here.
                        # But we can assume that upstream roads are independent.
                        # Thus, the sum of marginal entropies could be used here.
                        h_parents = sum(
                            self.calculate_gmm_entropy_approximation(self.gmm_per_segment[p]) for p in parents
                        )
                    conditional_entropy = h_joint - h_parents
                    total_entropy += conditional_entropy
                    print(f"Node {node}: H({node}|{parents}) = {conditional_entropy:.4f}")
                else:
                    raise ValueError('Missing join dist for calculate conditional entropy.')

        print(f"Total Entropy of the Bayesian Network: {total_entropy:.4f}")
        return total_entropy

