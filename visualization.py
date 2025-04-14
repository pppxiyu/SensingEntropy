from matplotlib import pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns


def _road_geo_get_direction(line):
    import numpy as np
    if line is None or line.is_empty:
        return 'unknown'

    start_x, start_y = line.coords[0]
    end_x, end_y = line.coords[-1]

    dx = end_x - start_x
    dy = end_y - start_y

    angle = np.degrees(np.arctan2(dy, dx))
    angle = (angle + 360) % 360  # Normalize to [0, 360)

    if 45 <= angle < 135:
        return 'northward'
    elif 135 <= angle < 225:
        return 'westward'
    elif 225 <= angle < 315:
        return 'southward'
    else:
        return 'eastward'


def _road_geo_directional_shift(geom, direction, offset=0.001):
    from shapely.affinity import translate
    if direction == 'eastward':
        return translate(geom, xoff=0, yoff=offset)
    elif direction == 'westward':
        return translate(geom, xoff=0, yoff=-offset)
    elif direction == 'northward':
        return translate(geom, xoff=offset, yoff=0)
    elif direction == 'southward':
        return translate(geom, xoff=-offset, yoff=0)
    else:
        return geom  # no shift for unknown direction


def map_roads_n_topology_plt(
    geo_roads=None,
    network=None, network_geo_roads=None,
    local_crs="EPSG:2263", shift=True, city_shp_path='./data/nybb_25a/nybb.shp', offset=0.0015
):
    import geopandas as gpd
    import networkx as nx

    # base map
    geo_city = gpd.read_file(city_shp_path).to_crs(local_crs)
    fig, ax = plt.subplots(figsize=(10, 10))
    geo_city.plot(ax=ax, facecolor='lightgray', edgecolor='none')

    # roads
    if geo_roads is not None:
        roads = geo_roads.copy()
        if shift:
            roads['direction'] = roads['geometry'].apply(_road_geo_get_direction)
            roads['geometry'] = roads.apply(
                lambda row: _road_geo_directional_shift(row['geometry'], row['direction'], offset=offset),
                axis=1
            )
        roads.to_crs(local_crs).plot(ax=ax, color='#3C535B', linewidth=1.5)

    # topology
    if network:
        roads_2 = network_geo_roads.copy()
        if shift:
            roads_2['direction'] = roads_2['geometry'].apply(_road_geo_get_direction)
            roads_2['geometry'] = roads_2.apply(
                lambda row: _road_geo_directional_shift(row['geometry'], row['direction'], offset=offset),
                axis=1
            )

        roads_2 = roads_2.to_crs(local_crs)
        roads_2["x"] = roads_2.geometry.centroid.x
        roads_2["y"] = roads_2.geometry.centroid.y
        node_positions = {row['link_id']: (row['x'], row['y']) for _, row in roads_2.iterrows()}
        valid_nodes = set(node_positions.keys())
        filtered_edges = [(u, v) for u, v in network.copy().edges() if u in valid_nodes and v in valid_nodes]

        graph = nx.DiGraph()
        graph.add_edges_from(filtered_edges)
        nx.draw(
            graph, pos=node_positions, ax=ax,
            node_size=10, node_color="#D95E32", edge_color="#D95E32", width=0.5, font_size=1
        )

    # show
    ax.set_axis_off()
    ax.set_xlim(geo_city.total_bounds[0], geo_city.total_bounds[2])
    ax.set_ylim(geo_city.total_bounds[1], geo_city.total_bounds[3])
    plt.show()


def map_roads_n_flood_plt(geo_roads, geo_flood, buffer=25):
    from shapely.ops import unary_union
    import geopandas as gpd
    geo_flood_filtered = geo_flood[geo_flood.geometry.within(unary_union(geo_roads.buffer(buffer).geometry))]
    fig, ax = plt.subplots(figsize=(8, 6))
    geo_roads.buffer(buffer).plot(ax=ax, color='blue', label='Roads')
    geo_flood_filtered.plot(ax=ax, color='red', markersize=1, label='Sensors')
    plt.show()
    return


def map_roads_n_flood_plotly(geo_roads, geo_flood, buffer=25, mapbox_token=None, save_dir='', shift=True):
    assert mapbox_token is not None, 'Missing mapbox token.'
    from shapely.ops import unary_union
    geo_flood_filtered = geo_flood[geo_flood.geometry.within(unary_union(geo_roads.buffer(buffer).geometry))]
    geo_flood_filtered = geo_flood_filtered.to_crs(epsg=4326)
    geo_roads = geo_roads.to_crs(epsg=4326)

    if shift:
        geo_roads['direction'] = geo_roads['geometry'].apply(_road_geo_get_direction)
        geo_roads['geometry'] = geo_roads.apply(
            lambda row: _road_geo_directional_shift(row['geometry'], row['direction']), axis=1
        )

    line_traces = []
    for _, row in geo_roads.iterrows():
        line_coords = list(row.geometry.coords)
        lons, lats = zip(*line_coords)
        line_traces.append(
            go.Scattermapbox(
                lon=lons, lat=lats, mode="lines", line=dict(width=2, color="blue"),
            )
        )
    point_traces = go.Scattermapbox(
        lon=geo_flood_filtered.geometry.x,
        lat=geo_flood_filtered.geometry.y,
        mode="markers", marker=dict(size=8, color="red"),
    )
    fig = go.Figure(line_traces + [point_traces])
    fig.update_layout(
        mapbox=dict(
            accesstoken=mapbox_token, style="light", zoom=9.5,
            center=dict(lat=geo_flood_filtered.geometry.y.mean(), lon=geo_flood_filtered.geometry.x.mean())
        ),
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        showlegend=False,
    )
    fig.show(renderer="browser")
    # fig.write_html(f'{save_dir}/dist_rc_error.html')
    # io.write_image(fig, f'{save_dir}/dist_rc_error.png', scale=4)
    return


def map_flood_p(gdf, p, mapbox_token, local_crs):
    import pandas as pd
    gdf = gdf.to_crs(local_crs)
    df = gdf.merge(p, on='link_id', how="left")
    p_lower = df['p'].quantile(0.05)  # 5th percentile
    p_upper = df['p'].quantile(0.95)  # 95th percentile
    df['p_clipped'] = df['p'].clip(lower=p_lower, upper=p_upper)
    p_min = df['p_clipped'].min()
    p_max = df['p_clipped'].max()
    df['p_norm'] = (df['p_clipped'] - p_min) / (p_max - p_min)

    fig = go.Figure()
    for _, row in df.iterrows():
        coords = list(row['geometry'].coords)  # Extract coordinates
        lons, lats = zip(*coords)  # Unpack into lat/lon
        if pd.isna(row['p']):
            color = "rgba(128, 128, 128, 0.8)"
        else:
            r = int(255 * row['p_norm'])
            g = int(255 * (1 - row['p_norm']))
            color = f"rgba({r}, {g}, 0, 0.8)"
        fig.add_trace(go.Scattermapbox(
            lon=lons,
            lat=lats,
            mode="lines",
            line=dict(width=3, color=color),
            hoverinfo="text",
            text=f"p: {row['p']:.4f}" if not pd.isna(row['p']) else "p: NaN"
        ))
    fig.update_layout(
        mapbox=dict(
            style="light", zoom=10, accesstoken=mapbox_token,
            center=dict(lat=df.geometry.centroid.y.mean(), lon=df.geometry.centroid.x.mean())
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False
    )
    fig.show(renderer="browser")


def dist_gmm_1d(gmm, speed_range=None,):
    import numpy as np
    from scipy.stats import norm
    if speed_range is None:
        speed_range = [-1, 70]

    plt.figure(figsize=(8, 5))
    speed_range = np.linspace(0, speed_range[1], 500)
    gmm_pdf = np.exp(gmm.score_samples(speed_range.reshape(-1, 1)))
    plt.plot(speed_range, gmm_pdf, label="GMM Fit", linewidth=2, color='red')
    weights = gmm.weights_
    means = gmm.means_.flatten()
    covariances = gmm.covariances_.flatten()
    for weight, mean, cov in zip(weights, means, covariances):
        component_pdf = weight * norm.pdf(speed_range, mean, np.sqrt(cov))
        plt.plot(speed_range, component_pdf, linestyle='dashed', label=f"Component μ={mean:.2f}", alpha=0.7)

    plt.xlabel("Traffic Speed")
    plt.ylabel("Density")
    plt.legend()
    plt.show()
    return


def dist_histo_gmm_1d(speeds, gmm, bins=30):
    import numpy as np
    from scipy.stats import norm

    plt.figure(figsize=(8, 5))
    sns.histplot(speeds, bins=bins, kde=False, stat='density', alpha=0.6, color="blue", label="Observed Data")

    speed_range = np.linspace(0, max(speeds), 500)
    gmm_pdf = np.exp(gmm.score_samples(speed_range.reshape(-1, 1)))
    plt.plot(speed_range, gmm_pdf, label="GMM Fit", linewidth=2, color='red')

    weights = gmm.weights_
    means = gmm.means_.flatten()
    covariances = gmm.covariances_.flatten()
    for weight, mean, cov in zip(weights, means, covariances):
        component_pdf = weight * norm.pdf(speed_range, mean, np.sqrt(cov))
        plt.plot(speed_range, component_pdf, linestyle='dashed', label=f"Component μ={mean:.2f}", alpha=0.7)

    plt.xlabel("Traffic Speed")
    plt.ylabel("Density")
    plt.legend()
    plt.show()
    return


def dist_gmm_3d(gmm, segment1=None, segment2=None, speed_range=(-10, 90)):
    import numpy as np
    from scipy.stats import multivariate_normal
    x = np.linspace(speed_range[0], speed_range[1], 100)
    y = np.linspace(speed_range[0], speed_range[1], 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    for mean, cov, weight in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        rv = multivariate_normal(mean=mean, cov=cov)
        grid_points = np.dstack((X, Y))
        Z += weight * rv.pdf(grid_points)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.9)
    ax.set_xlabel(f'Traffic Speed on {segment1}')
    ax.set_ylabel(f'Traffic Speed on {segment2}')
    plt.show()


def dist_discrete_gmm(dist):
    import numpy as np
    from scipy.stats import norm

    p_flood = dist['p_flood']
    p_no_flood = 1 - p_flood
    gmm_no_flood = dist['speed_no_flood']
    gmm_flood = dist['speed_flood']
    speed_range = np.linspace(-1, 80, 1000)

    def gmm_pdf(gmm, x):
        if gmm is None:
            return np.zeros_like(x, dtype=float)
        pdf = np.zeros_like(x, dtype=float)
        for weight, mean, cov in zip(gmm.weights_, gmm.means_.flatten(), gmm.covariances_.flatten()):
            pdf += weight * norm.pdf(x, loc=mean, scale=np.sqrt(cov))
        return pdf

    pdf_no_flood = gmm_pdf(gmm_no_flood, speed_range)
    pdf_flood = gmm_pdf(gmm_flood, speed_range)

    plt.figure(figsize=(10, 6))
    plt.plot(speed_range, pdf_no_flood, label=f'No Flood (P={p_no_flood:.3f})', color='blue')
    plt.plot(speed_range, pdf_flood, label=f'Flood (P={p_flood:.3f})', color='red')
    plt.fill_between(speed_range.flatten(), pdf_no_flood, alpha=0.2, color='blue')
    plt.fill_between(speed_range.flatten(), pdf_flood, alpha=0.2, color='red')
    plt.xlabel('Speed')
    plt.ylabel('Probability Density (Scaled by P(I))')
    plt.legend()
    plt.show()
    pass


def bar_closure_count_over_time(df, ):
    df.set_index('time_create', inplace=True)
    interval = '1D'
    time_counts = df.resample(interval).size().fillna(0)

    plt.figure(figsize=(10, 5))
    plt.plot(time_counts.index, time_counts.values, marker='o', linestyle='-')
    plt.xlabel('Time')
    plt.ylabel('Closure count')
    plt.xticks(rotation=45)
    plt.grid()
    plt.show()


def bar_flood_prob(row):
    p = row['p']
    not_p = 1 - p

    categories = ['Flooded', 'Not Flooded']
    values = [p, not_p]

    plt.figure(figsize=(8, 6))
    bars = plt.bar(categories, values, color=['blue', 'green'])
    plt.title('Probability of Flooding', fontsize=12, pad=15)
    plt.xlabel('Condition', fontsize=10)
    plt.ylabel('Probability', fontsize=10)
    plt.ylim(0, 1)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.4f}',
                 ha='center', va='bottom')

    plt.tight_layout()
    plt.show()


def heatmap_corr(matrix):
    import seaborn as sns
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=False, fmt=".2f", cmap="coolwarm", linewidths=0.5)
    plt.show()

