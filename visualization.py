from matplotlib import pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns


def overlap_roads_flood_plt(geo_roads, geo_flood, buffer=25):
    from shapely.ops import unary_union
    geo_flood_filtered = geo_flood[geo_flood.geometry.within(unary_union(geo_roads.buffer(buffer).geometry))]
    fig, ax = plt.subplots(figsize=(8, 6))
    geo_roads.buffer(buffer).plot(ax=ax, color='blue', label='Roads')
    geo_flood_filtered.plot(ax=ax, color='red', markersize=1, label='Sensors')
    plt.show()
    return


def overlap_roads_flood_plotly(geo_roads, geo_flood, buffer=25, mapbox_token=None, save_dir=''):
    assert mapbox_token is not None, 'Missing mapbox token.'
    from shapely.ops import unary_union
    geo_flood_filtered = geo_flood[geo_flood.geometry.within(unary_union(geo_roads.buffer(buffer).geometry))]
    geo_flood_filtered = geo_flood_filtered.to_crs(epsg=4326)
    geo_roads = geo_roads.to_crs(epsg=4326)

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
        showlegend=False, dragmode='zoom',
    )
    fig.show(renderer="browser")
    # fig.write_html(f'{save_dir}/dist_rc_error.html')
    # io.write_image(fig, f'{save_dir}/dist_rc_error.png', scale=4)
    return


def dist_gmm_1d(speeds, gmm, bins=30):
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


def heatmap_corr(matrix):
    import seaborn as sns
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=False, fmt=".2f", cmap="coolwarm", linewidths=0.5)
    plt.show()


def map_road_network_connections(gdf, network, local_crs):
    import networkx as nx
    gdf = gdf.to_crs(local_crs)
    gdf["x"] = gdf.geometry.centroid.x
    gdf["y"] = gdf.geometry.centroid.y
    node_positions = {row['link_id']: (row['x'], row['y']) for _, row in gdf.iterrows()}

    valid_segments = set(gdf["link_id"])
    filtered_edges = [(u, v) for u, v in network.edges() if u in valid_segments and v in valid_segments]
    graph = nx.DiGraph()
    graph.add_edges_from(filtered_edges)

    fig, ax = plt.subplots(figsize=(15, 15))
    gdf.plot(ax=ax, color="red", linewidth=1, alpha=0.5)
    nx.draw(
        graph,
        pos=node_positions,
        ax=ax,
        with_labels=True,
        node_size=10,
        node_color="grey",
        edge_color="blue",
        width=.25,
        font_size=1
    )
    plt.show()


def dist_gmm_3d(gmm, segment1, segment2, speed_range=(-10, 90)):
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



