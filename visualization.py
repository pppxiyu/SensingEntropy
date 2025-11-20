from matplotlib import pyplot as plt
import os


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


def legacy_map_roads_n_flood_plotly(geo_roads, geo_flood, buffer=25, mapbox_token=None, save_dir='', shift=True):
    assert mapbox_token is not None, 'Missing mapbox mb_token.'
    from shapely.ops import unary_union
    import plotly.graph_objects as go
    import plotly.express as px
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


def legacy_map_roads_plotly(geo_roads, mapbox_token=None, save_dir='',):
    import plotly.graph_objects as go
    import plotly.express as px

    assert mapbox_token is not None, 'Missing mapbox mb_token.'
    geo_roads = geo_roads.to_crs(epsg=4326)

    line_traces = []
    legend_added = set()
    for _, row in geo_roads.iterrows():

        if row.geometry.geom_type == 'LineString':
            linestrings = [row.geometry]
        else:
            linestrings = list(row.geometry.geoms)

        for linestring in linestrings:
            line_coords = list(linestring.coords)
            lons, lats = zip(*line_coords)

            p_value = row['p']
            if p_value == 'Reported':
                color = "#DE3163"
                display_p = "Reported"
                legend_name = "Reported"
                legend_group = "Reported"
            elif p_value == 'Unknown':
                color = "#808080"
                display_p = "Unknown"
                legend_name = "Unknown"
                legend_group = "Unknown"
            elif p_value >= 0.666666:
                color = "red"
                display_p = f"{p_value:.3f}"
                legend_name = "High Risk (p ≥ 0.667)"
                legend_group = "high"
            elif p_value >= 0.333333:
                color = "orange"
                display_p = f"{p_value:.3f}"
                legend_name = "Medium Risk (0.333 ≤ p < 0.667)"
                legend_group = "medium"
            else:
                color = "yellow"
                display_p = f"{p_value:.3f}"
                legend_name = "Low Risk (p < 0.333)"
                legend_group = "low"

            show_legend = legend_group not in legend_added
            if show_legend:
                legend_added.add(legend_group)

            line_traces.append(
                go.Scattermapbox(
                    lon=lons,
                    lat=lats,
                    mode="lines",
                    line=dict(width=3, color=color),
                    # name=f"{row['STREET']} (p={display_p})",
                    name=legend_name,
                    legendgroup=legend_group,
                    showlegend=show_legend,
                    hovertemplate=f"<b>{row['STREET']}</b><br>Probability: {display_p}<extra></extra>",
                )
            )

    fig = go.Figure(line_traces)
    fig.update_layout(
        width=2000,
        height=800,
        mapbox=dict(
            accesstoken=mapbox_token, style="dark", center=dict(lat=32.79, lon=-79.94), zoom=13.5,
        ),
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        showlegend=True,
        dragmode='pan',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1,
            font=dict(size=20)
        ),
    )

    fig.show(renderer="browser",)
    fig.write_image("map_plot.pdf",)
    # fig.write_html('bn_outputs.html')
    # fig.write_html(f'{save_dir}/bn_outputs.html')
    # io.write_image(fig, f'{save_dir}/dist_rc_error.png', scale=4)
    return


def legacy_dist_histo_gmm_1d(speeds, gmm, bins=30, save_dir=None):
    import numpy as np
    from scipy.stats import norm
    import seaborn as sns

    fig = plt.figure(figsize=(8, 5))
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

    if save_dir is not None:
        os.makedirs(os.path.dirname(save_dir) or '.', exist_ok=True)
        fig.savefig(
            os.path.join(save_dir),
            dpi=300, 
            edgecolor='none',      # No edge color
            transparent=True
        )
        print(f'Saved scatter plot to {save_dir}')
    else:
        plt.show()

    return


def map_roads_n_topology(
    geo_roads=None,
    network=None, network_geo_roads=None,
    local_crs="EPSG:2263", shift=True, city_shp_path=None, offset=0.0015, save_dir=None
):
    import geopandas as gpd
    import networkx as nx
    import os

    fig, ax = plt.subplots(figsize=(10, 10))

    # base map
    if city_shp_path is not None:
        geo_city = gpd.read_file(city_shp_path).to_crs(local_crs)
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

    if save_dir is not None:
        os.makedirs(os.path.dirname(save_dir) or '.', exist_ok=True)
        fig.savefig(
            os.path.join(save_dir),
            dpi=600, 
            bbox_inches='tight',  # Remove extra whitespace
            # facecolor='white',     # Background color
            edgecolor='none',      # No edge color
            transparent=True
        )
        print(f'Saved scatter plot to {save_dir}')
    else:
        plt.show()
    
    return


def map_roads_n_flood(geo_roads, geo_flood, buffer=25):
    from shapely.ops import unary_union
    import geopandas as gpd
    geo_flood_filtered = geo_flood[geo_flood.geometry.within(unary_union(geo_roads.buffer(buffer).geometry))]
    fig, ax = plt.subplots(figsize=(8, 6))
    geo_roads.buffer(buffer).plot(ax=ax, color='blue', label='Roads')
    geo_flood_filtered.plot(ax=ax, color='red', markersize=1, label='Sensors')
    plt.show()
    return


def map_roads_n_values(
        geo_roads, values, local_crs="EPSG:2263",
        shift=True, city_shp_path=None, offset=0.0015,
        y_label='', save_dir=None, coord_decimals=3,
        cmap="viridis", max_ticks=6, enlarge_font=True,
        close_axis=False,
):
    import pandas as pd
    import geopandas as gpd
    import os
    from matplotlib.ticker import FuncFormatter, MaxNLocator
    from pyproj import Transformer
    import matplotlib as mpl
    try:
        import cmasher as cmr
    except ImportError:
        pass

    fig, ax = plt.subplots(figsize=(10, 10))

    # base map
    if city_shp_path is not None:
        geo_city = gpd.read_file(city_shp_path).to_crs(local_crs)
        geo_city.plot(ax=ax, facecolor='none', edgecolor='lightgrey', linewidth=1.5)

    # roads
    roads = geo_roads.copy()
    if shift:
        roads['direction'] = roads['geometry'].apply(_road_geo_get_direction)
        roads['geometry'] = roads.apply(
            lambda row: _road_geo_directional_shift(row['geometry'], row['direction'], offset=offset),
            axis=1
        )
    roads["value"] = roads['link_id'].map(values)
    
    # Handle cmasher colormaps
    cmap_to_use = cmap
    if isinstance(cmap, str) and not cmap.startswith('cmr.'):
        # Check if it's a cmasher colormap (without 'cmr.' prefix)
        try:
            cmap_to_use = mpl.colormaps[f'cmr.{cmap}']
        except (KeyError, AttributeError):
            # If not found in cmasher, use as is (standard matplotlib colormap)
            cmap_to_use = cmap
    
    roads_with_data = roads[pd.notna(roads['value'])]
    roads_no_data = roads[pd.isna(roads['value'])]
    roads_no_data.to_crs(local_crs).plot(ax=ax, color="#8d8d8d", linewidth=1.5,label="no data", )
    roads_with_data.to_crs(local_crs).plot(
        ax=ax, column="value", cmap=cmap_to_use, linewidth=5.0, legend=True, legend_kwds={"shrink": 0.6},
    )

    # show
    ax.set_xlim(geo_city.total_bounds[0], geo_city.total_bounds[2])
    ax.set_ylim(geo_city.total_bounds[1], geo_city.total_bounds[3])

    # Make the box thicker
    for spine in ax.spines.values():
        spine.set_linewidth(2.0)  # Thicker box

    # Add padding between axis and plot (as a percentage of the data range)
    x_range = geo_city.total_bounds[2] - geo_city.total_bounds[0]
    y_range = geo_city.total_bounds[3] - geo_city.total_bounds[1]
    padding = 0.02  # 2% padding

    ax.set_xlim(geo_city.total_bounds[0] - x_range * padding, 
                geo_city.total_bounds[2] + x_range * padding)
    ax.set_ylim(geo_city.total_bounds[1] - y_range * padding, 
                geo_city.total_bounds[3] + y_range * padding)

    # Create transformer for coordinate conversion
    transformer = Transformer.from_crs(local_crs, "EPSG:4326", always_xy=True)

    # Convert axis to longitude/latitude (WGS84)
    def x_formatter(x, pos):
        lon, _ = transformer.transform(x, geo_city.total_bounds[1])
        return f'{lon:.{coord_decimals}f}°'

    def y_formatter(y, pos):
        _, lat = transformer.transform(geo_city.total_bounds[0], y)
        return f'{lat:.{coord_decimals}f}°'

    # Limit number of ticks
    ax.xaxis.set_major_locator(MaxNLocator(nbins=max_ticks, prune='both'))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=max_ticks, prune='both'))
    
    ax.xaxis.set_major_formatter(FuncFormatter(x_formatter))
    ax.yaxis.set_major_formatter(FuncFormatter(y_formatter))

    # Font sizes based on enlarge_font parameter
    tick_labelsize = 24 if enlarge_font else 12
    axis_labelsize = 28 if enlarge_font else 14
    colorbar_labelsize = 32 if enlarge_font else 16
    colorbar_ticksize = 24 if enlarge_font else 12
    tick_width = 2.5 if enlarge_font else 1.0
    tick_length = 10 if enlarge_font else 6

    # Enlarge tick labels and add axis labels
    ax.tick_params(axis='both', which='major', labelsize=tick_labelsize)
    ax.set_xlabel('Longitude', fontsize=axis_labelsize)
    ax.set_ylabel('Latitude', fontsize=axis_labelsize)

    # Handle close_axis option - make ticks and labels white
    if close_axis:
        ax.tick_params(axis='both', colors='white', labelcolor='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')

    # Colorbar styling with larger fonts
    cax = ax.get_figure().axes[-1]
    for spine in cax.spines.values():
        spine.set_visible(False)
    cax.tick_params(labelsize=colorbar_ticksize)
    cax.set_ylabel(y_label, fontsize=colorbar_labelsize)

    ax.tick_params(axis='both', which='major', labelsize=tick_labelsize, width=tick_width, length=tick_length)
    
    plt.tight_layout(pad=.5)

    if save_dir is not None:
        os.makedirs(os.path.dirname(save_dir) or '.', exist_ok=True)
        fig.savefig(
            os.path.join(save_dir),
            dpi=300, 
            bbox_inches='tight',  # Remove extra whitespace
            facecolor='white',     # Background color
            edgecolor='none',      # No edge color
            transparent=False
        )
        print(f'Saved scatter plot to {save_dir}')
    else:
        plt.show()

    return


def map_road_n_disruption(disruptions, roads_geo, city_shp_path, local_crs="EPSG:2263", shift=False, offset=0.0015):
    import geopandas as gpd
    from collections import defaultdict
    import numpy as np

    # convert the format of disruptions
    # Initialize dictionaries to store all values for each road
    metric1_dict = defaultdict(list)
    metric2_dict = defaultdict(list)
    # Iterate through each element in disruptions
    for item in disruptions:
        # Skip if the item has None values
        if item[0] is None or item[1] is None:
            continue
        metric1_values = item[0]  # List of metric_1 values
        metric2_values = item[1]  # List of metric_2 values
        roads = item[2]  # List of road IDs
        # Add each road's metrics to the dictionaries
        for i, road in enumerate(roads):
            metric1_dict[road].append(metric1_values[i])
            metric2_dict[road].append(metric2_values[i])
    # Calculate mean for roads with multiple values
    metric1_final = {road: np.mean(values) for road, values in metric1_dict.items()}
    metric2_final = {road: np.mean(values) for road, values in metric2_dict.items()}

    for values in [metric1_final, metric2_final]:
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # base map
        if city_shp_path is not None:
            geo_city = gpd.read_file(city_shp_path).to_crs(local_crs)
            geo_city.plot(ax=ax, facecolor='lightgray', edgecolor='none')

        # roads
        roads = roads_geo.copy()
        if shift:
            roads['direction'] = roads['geometry'].apply(_road_geo_get_direction)
            roads['geometry'] = roads.apply(
                lambda row: _road_geo_directional_shift(row['geometry'], row['direction'], offset=offset),
                axis=1
            )

        roads["value"] = roads['link_id'].map(values)
        roads = roads[~roads["value"].isna()]
        roads.to_crs(local_crs).plot(
            ax=ax, column="value", cmap="viridis", linewidth=1.5, legend=True,
            missing_kwds={"color": "#c5c5c5", "label": "no data"},
            legend_kwds={"shrink": 0.6,},
        )

        # show
        ax.set_axis_off()
        ax.set_xlim(geo_city.total_bounds[0], geo_city.total_bounds[2])
        ax.set_ylim(geo_city.total_bounds[1], geo_city.total_bounds[3])
        cax = ax.get_figure().axes[-1]
        for spine in cax.spines.values():
            spine.set_visible(False)
        cax.tick_params(labelsize=16)
        cax.set_ylabel("Network entropy", fontsize=18,)
        plt.tight_layout(pad=.5)
        plt.show()

    return


def dist_gmm_1d(gmm, speed_range=None, title=None, save_dir=None, return_y_max=False, y_limit=None):
    import numpy as np
    from scipy.stats import norm
    if speed_range is None:
        speed_range = [-1, 80]

    fig, ax = plt.subplots(figsize=(5, 5))  # Square figure
    speed_range_2 = np.linspace(0, speed_range[1], 500)
    gmm_pdf = np.exp(gmm.score_samples(speed_range_2.reshape(-1, 1)))
    ax.plot(speed_range_2, gmm_pdf, label="GMM", linewidth=2, color='#743B8C')  # Changed color

    weights = gmm.weights_
    means = gmm.means_.flatten()
    covariances = gmm.covariances_.flatten()
    for i, (weight, mean, cov) in enumerate(zip(weights, means, covariances)):
        component_pdf = weight * norm.pdf(speed_range_2, mean, np.sqrt(cov))
        # Add label only for the first component
        label = "Components" if i == 0 else None
        ax.plot(speed_range_2, component_pdf, linestyle='dashed', color='grey', alpha=0.7, label=label)

    ax.set_xlabel("Averaged speed (mph)", fontsize=24)  # Enlarged font
    ax.set_ylabel("Density", fontsize=24)  # Enlarged font
    ax.tick_params(axis='both', which='major', labelsize=14)  # Enlarged tick labels
    ax.legend(frameon=False, fontsize=16)  # No box, larger font

    # Set y-axis to start from 0
    ax.set_ylim(bottom=0)
    if y_limit is not None:
        ax.set_ylim(0, y_limit)

    plt.tight_layout()
    ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')  # Make plot area square

    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    if save_dir is not None:
        os.makedirs(os.path.dirname(save_dir) or '.', exist_ok=True)
        fig.savefig(
            os.path.join(save_dir),
            dpi=300, 
            bbox_inches='tight',
            edgecolor='none',
            transparent=True
        )
        print(f'Saved plot to {save_dir}')
    else:
        plt.show()

    if return_y_max:
        return ax.get_ylim()[1]


def dist_gmm_3d(
        gmm, segment1=None, segment2=None, 
        speed_range=(0, 90), z_limit=None, return_z_max=False, save_dir=None,
        switch_axis=False
):
    import numpy as np
    from scipy.stats import multivariate_normal

    def _truncate_colormap(cmap_name='viridis', minval=0.2, maxval=0.8, n=256):
        import numpy as np
        from matplotlib.colors import LinearSegmentedColormap
        base_cmap = plt.get_cmap(cmap_name)
        new_cmap = LinearSegmentedColormap.from_list(
            f'{cmap_name}_trunc',
            base_cmap(np.linspace(minval, maxval, n))
        )
        return new_cmap

    x = np.linspace(speed_range[0], speed_range[1], 100)
    y = np.linspace(speed_range[0], speed_range[1], 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    for mean, cov, weight in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        rv = multivariate_normal(mean=mean, cov=cov)
        grid_points = np.dstack((X, Y))
        Z += weight * rv.pdf(grid_points)

    if switch_axis:
        Z = Z.T
        segment1, segment2 = segment2, segment1

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(
        X, Y, Z,
        cmap=_truncate_colormap('viridis', 0.15, 1),
        edgecolor='none', antialiased=False, alpha=1
    )

    ax.set_xlabel(f'Speed on Link\n{segment1} (mph)', fontsize=28, labelpad=30)
    ax.set_ylabel(f'Link {segment2}', fontsize=28, labelpad=22)
    ax.set_zlabel('Density', fontsize=28, labelpad=16)  # labels
    ax.ticklabel_format(axis='z', style='sci', scilimits=(-2, 2))
    ax.zaxis.get_offset_text().set_fontsize(16)  # scientific notation

    if z_limit is not None:
        ax.set_zlim(Z.min(), z_limit)
    else:
        ax.set_zlim(Z.min(),)
    ax.set_xlim(x.min(), x.max())
    ax.set_xlim(ax.get_xlim()[::-1])
    ax.set_ylim(y.min(), y.max())  # tight config

    ax.grid(False)  # remove grid

    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.set_edgecolor('black')
        axis.pane.set_linewidth(1.5)  # box edges
        axis.pane.fill = False  # bg color
        axis._axinfo['tick']['inward_factor'] = 0
        axis._axinfo['tick']['outward_factor'] = 0  # remove ticks

    ax.tick_params(axis='x', labelsize=23)
    ax.tick_params(axis='y', labelsize=23)
    ax.tick_params(axis='z', labelsize=19)  # font size

    xs = [ax.get_xlim()[0], ax.get_xlim()[0]]
    ys = [ax.get_ylim()[0], ax.get_ylim()[0]]
    zs = [ax.get_zlim()[0], ax.get_zlim()[1]]
    ax.plot(xs, ys, zs, color='black', linewidth=1)  # verticle line on the left

    xticks = np.linspace(x.min(), x.max(), 7)
    yticks = np.linspace(y.min(), y.max(), 7)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)  # spase label
    
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.12)  # Increase bottom margin

    if save_dir is not None:
        os.makedirs(os.path.dirname(save_dir) or '.', exist_ok=True)
        fig.savefig(
            os.path.join(save_dir),
            dpi=300, 
            edgecolor='none',      # No edge color
            transparent=True
        )
        print(f'Saved scatter plot to {save_dir}')
    else:
        plt.show()

    if return_z_max:
        return ax.get_zlim()[1]


def dist_discrete_gmm(dist_w_obs, gmm_wo_abs=None, close_curve=[], save_dir=None):
    import numpy as np
    from scipy.stats import norm
    import matplotlib.ticker as mtick

    p_flood = dist_w_obs['p_flood']
    p_no_flood = 1 - p_flood
    gmm_no_flood = dist_w_obs['speed_no_flood']
    gmm_flood = dist_w_obs['speed_flood']
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
    if gmm_wo_abs is not None:
        pdf_wo_abs = gmm_pdf(gmm_wo_abs, speed_range)

    fig, ax = plt.subplots(figsize=(5, 5))  # Square figure to match

    if gmm_wo_abs is not None:
        ax.plot(speed_range, pdf_wo_abs, label=f'w/o observation', color='grey', linestyle='--', linewidth=2)
        # ax.fill_between(speed_range, pdf_wo_abs, alpha=0.2, color='grey')

    if 'no_flood' not in close_curve:
        ax.plot(speed_range, pdf_no_flood, label=f'No Flood (p={p_no_flood:.3f})', color='#6F8ABF', linewidth=2)
        ax.fill_between(speed_range, pdf_no_flood, alpha=0.2, color='#DFEBF2')

    if 'flood' not in close_curve:
        ax.plot(speed_range, pdf_flood, label=f'Flood (p={p_flood:.3f})', color='#743B8C', linewidth=2)
        ax.fill_between(speed_range, pdf_flood, alpha=0.2, color="#ADB8D9")

    # Match font sizes from dist_gmm_1d
    ax.set_xlabel('Averaged speed (mph)', fontsize=24)
    ax.set_ylabel('Density', fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=14)

    # Scientific notation formatting
    ax.yaxis.set_major_formatter(mtick.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

    # Match legend style
    ax.legend(frameon=False, fontsize=16)

    # Match spine thickness
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')  # Make plot area square

    if save_dir is not None:
        os.makedirs(os.path.dirname(save_dir) or '.', exist_ok=True)
        fig.savefig(
            os.path.join(save_dir),
            dpi=300, 
            bbox_inches='tight',
            edgecolor='none',
            transparent=True
        )
        print(f'Saved plot to {save_dir}')
    else:
        plt.show()

    return


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

    categories = ['Flood', 'No Flood']
    values = [p, not_p]

    plt.rc('font', size=28)
    plt.figure(figsize=(8, 6))
    bars = plt.bar(categories, values, color=['#C4303E', '#006AAE'])
    plt.ylabel('Probability')
    plt.ylim(0, 1)

    for bar, va, of, c in zip(bars, ['bottom', 'top'], [0, -0.02], ['black', 'white']):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + of,f'{height:.4f}', ha='center', va=va, color=c)

    plt.tight_layout()
    plt.show()


def scatter_diff_vs_estimated_diff(
        kld, kld_estimated, out_file=None, 
        xscale='log', yscale='log', reg='linear', base=10,
        save_dir=None, x_title='KLD1', y_title='KLD2', if_norm=False
    ):
    import numpy as np
    import os
    from scipy import stats
    from matplotlib.ticker import LogLocator

    if xscale not in ('linear', 'log',):
        raise ValueError("xscale must be one of 'linear', or 'log'")
    if yscale not in ('linear', 'log',):
        raise ValueError("xscale must be one of 'linear', or 'log'")
    if reg not in ('robust', 'linear',):
        raise ValueError("xscale must be one of 'robust', or 'linear'")
    
    x = np.asarray(kld)
    y = np.asarray(kld_estimated)

    # norm
    if if_norm:
        # min_v = min(np.nanmin(x), np.nanmin(y))
        # max_v = max(np.nanmax(x), np.nanmax(y))
        x = (x - x.min()) * (1 - 2 * 0.025)/ (x.max() - x.min()) + 0.025
        y = (y - y.min()) * (1 - 2 * 0.025) / (y.max() - y.min()) + 0.025

    # regression line
    def _txy(scale, v):
        if scale == 'linear':
            return v
        if scale == 'log':
            return np.log(v) / np.log(base)
        
    def _inv_txy(scale, v):
        if scale == 'linear':
            return v
        if scale == 'log':
            return base ** v
        
    tx = _txy(xscale, x)
    ty = _txy(yscale, y)
    if reg == 'robust':
        slope, intercept, *_ = stats.theilslopes(ty, tx, 0.95)
    elif reg == 'linear':
        slope, intercept = np.polyfit(tx, ty, 1)
    xs = np.linspace(np.nanmin(x), np.nanmax(x), 400)
    y_line_t = slope * _txy(xscale, xs) + intercept
    y_line = _inv_txy(yscale, y_line_t)

    # calculate r2
    from sklearn.linear_model import LinearRegression
    model = LinearRegression().fit(y.reshape(-1, 1), x)
    r2 = model.score(y.reshape(-1, 1), x)

    # visualization
    fig, ax = plt.subplots(figsize=(6, 6))  # Square figure
    ax.scatter(x, y, alpha=0.85, s=60, marker='o', color='#11AEBF')  # Changed color
    ax.set_xlabel(x_title, fontsize=18)
    ax.set_ylabel(y_title, fontsize=18)

    if xscale == 'log':
        ax.set_xscale('log', base=base)
        ax.xaxis.set_major_locator(LogLocator(base=base, subs=[1.0]))  # Only show 10^n ticks
        ax.xaxis.set_minor_locator(LogLocator(base=base, subs=[1.0, 2.0, 5.0]))  # Show 2*10^n and 5*10^n as minor
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0e}'.replace('e-0', 'e-').replace('e+0', 'e+')))
        ax.xaxis.set_minor_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0e}'.replace('e-0', 'e-').replace('e+0', 'e+')))

    if yscale == 'log':
        ax.set_yscale('log', base=base)
        ax.yaxis.set_major_locator(LogLocator(base=base, subs=[1.0]))  # Only show 10^n ticks
        ax.yaxis.set_minor_locator(LogLocator(base=base, subs=[1.0, 2.0, 5.0]))  # Show 2*10^n and 5*10^n as minor
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0e}'.replace('e-0', 'e-').replace('e+0', 'e+')))
        ax.yaxis.set_minor_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0e}'.replace('e-0', 'e-').replace('e+0', 'e+')))

    # Increase tick label font size
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.tick_params(axis='both', which='minor', labelsize=14.5)

    # Make the box (spines) thicker
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    plt.tight_layout()
    ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')

    # Get axis limits after layout is set
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Extend the fitted line to touch the box boundaries
    # Calculate y values at the box boundaries using the fitted line equation
    y_at_xlim = [xlim[0] * (y_line[-1] - y_line[0]) / (xs[-1] - xs[0]) + (y_line[0] - xs[0] * (y_line[-1] - y_line[0]) / (xs[-1] - xs[0])),
                xlim[1] * (y_line[-1] - y_line[0]) / (xs[-1] - xs[0]) + (y_line[0] - xs[0] * (y_line[-1] - y_line[0]) / (xs[-1] - xs[0]))]

    ax.plot([xlim[0], xlim[1]], y_at_xlim, color='#743B8C', linestyle='--', linewidth=2, label='Fitted line', clip_on=False)

    # Create legend without frame and add R² as title with matching font sizes
    # Move to upper left and align left
    r2_text = f'$R^2$ = {r2:.3f}' if np.isfinite(r2) else '$R^2$ = N/A'
    legend = ax.legend(fontsize=16, frameon=False, loc='upper left', title=r2_text, title_fontsize=16, alignment='left')

    # Reset limits to ensure they don't change
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    if save_dir is not None:
        os.makedirs(os.path.dirname(save_dir) or '.', exist_ok=True)
        fig.savefig(os.path.join(save_dir))
        print(f'Saved scatter plot to {save_dir}')
    else:
        plt.show()

    return r2


def scatter_disp_n_supr_w_factors(
        x, y, xlabel="X", ylabel="Y", 
        dot_color='#8B6FBF', line_color='#4B3E73',
        save_dir=None, x_limit=None, y_limit=[-.05, 1.05],
        show_pvalue=False, legend_loc='upper left'
):
    import numpy as np
    from scipy import stats
    
    fig, ax = plt.subplots(figsize=(5.5, 5))  # Square figure to match
    
    # Scatter plot with new color - removed edges
    ax.scatter(x, y, c=dot_color, s=50, alpha=0.85, edgecolors='none')
    
    # Calculate trend line
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    
    ax.set_xlabel(xlabel, fontsize=20)  # Smaller font
    ax.set_ylabel(ylabel, fontsize=20)  # Smaller font
    ax.tick_params(axis='both', which='major', labelsize=14)  # Enlarged tick labels
    
    # Set axis limits if provided
    if x_limit is not None:
        ax.set_xlim(x_limit)
    if y_limit is not None:
        ax.set_ylim(y_limit)
    
    # Remove grid (matching dist_gmm_1d style - no grid)
    ax.grid(False)
    
    plt.tight_layout()
    
    # Adjust left margin to prevent y-axis label cutoff
    fig.subplots_adjust(left=0.18)
    
    ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')  # Make plot area square
    
    # Thicker spines to match
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    
    # Get axis limits after layout is set
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # Extend the fitted line to touch the box boundaries
    y_at_xlim = [slope * xlim[0] + intercept, slope * xlim[1] + intercept]
    
    # Plot dashed trend line touching the box
    ax.plot([xlim[0], xlim[1]], y_at_xlim, color=line_color, linestyle='--', linewidth=2, label='Fitted line', clip_on=False)
    
    # Only add p-value if show_pvalue is True
    if show_pvalue:
        # Add p-value annotation with italic p
        if p_value < 0.001:
            p_text = r'$\it{p}$-value < 0.001'
        else:
            p_text = r'$\it{p}$-value = ' + f'{p_value:.3f}'
        
        # Create legend with p-value as title in the specified location
        legend = ax.legend(fontsize=14, frameon=False, loc=legend_loc, 
                          title=p_text, title_fontsize=14, alignment='left',
                          facecolor='white', framealpha=0.9)
    else:
        # Create legend without p-value in the specified location
        legend = ax.legend(fontsize=14, frameon=False, loc=legend_loc, 
                          alignment='left')
    
    # Reset limits to ensure they don't change
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
    if save_dir is not None:
        os.makedirs(os.path.dirname(save_dir) or '.', exist_ok=True)
        fig.savefig(
            os.path.join(save_dir),
            dpi=300, 
            bbox_inches='tight',
            pad_inches=0.3,  # Increased padding for saving
            edgecolor='none',
            transparent=True
        )
        print(f'Saved plot to {save_dir}')
    else:
        plt.show()
    
    return


def line_placement_d_n_uod(
    y_left, y_right,
    xlabel="X", ylabel_left="Left Y", ylabel_right="Right Y",
    color_left="#8B6FBF", color_right='#6F8ABF',
    linestyle_left="-", linestyle_right="--",
    x_limit=None, y_limit_left=None, y_limit_right=None,
    legend_loc='lower left', save_dir=None, y_buffer=0.4
):
    """
    Plot two line curves with dual y-axes on a wide rectangular figure.
    
    Parameters:
    -----------
    y_left, y_right : array-like
        Data for left and right y-axes
    xlabel, ylabel_left, ylabel_right : str
        Axis labels
    color_left, color_right : str
        Line colors
    linestyle_left, linestyle_right : str
        Line styles
    x_limit, y_limit_left, y_limit_right : tuple or None
        Axis limits (min, max)
    legend_loc : str or int
        Legend position ('upper left', 'lower right', 'best', etc.)
    save_dir : str or None
        Path to save figure
    y_buffer : float
        Y-axis padding factor (default 0.4 = 40%)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    # Convert to numpy arrays
    y_left = np.array(y_left, dtype=float)
    y_right = np.array(y_right, dtype=float)

    # Ensure equal length
    if len(y_left) != len(y_right):
        raise ValueError("y_left and y_right must have the same length.")

    # Auto-generate x values
    x = np.arange(len(y_left)) + 1

    # Wide rectangular figure
    fig, ax_left = plt.subplots(figsize=(12, 3.75))
    ax_right = ax_left.twinx()

    # Plot both curves with markers
    line_left, = ax_left.plot(
        x, y_left, color=color_left, linestyle=linestyle_left,
        linewidth=2.5, marker='o', markersize=6, label=ylabel_left
    )
    line_right, = ax_right.plot(
        x, y_right, color=color_right, linestyle=linestyle_right,
        linewidth=2.5, marker='o', markersize=6, label=ylabel_right
    )

    # Axis labels
    ax_left.set_xlabel(xlabel, fontsize=20, color='black')
    ax_left.set_ylabel(ylabel_left, fontsize=20, color='black')
    ax_right.set_ylabel(ylabel_right, fontsize=20, color='black')

    # Tick label style
    ax_left.tick_params(axis='both', which='major', labelsize=14, colors='black')
    ax_right.tick_params(axis='both', which='major', labelsize=14, colors='black')

    # Axis limits with padding
    if x_limit is not None:
        ax_left.set_xlim(x_limit)

    if y_limit_left is None:
        y_min, y_max = np.min(y_left), np.max(y_left)
        pad = (y_max - y_min) * y_buffer or y_buffer
        ax_left.set_ylim(y_min - pad, y_max + pad)
    else:
        ax_left.set_ylim(y_limit_left)

    if y_limit_right is None:
        y_min, y_max = np.min(y_right), np.max(y_right)
        pad = (y_max - y_min) * y_buffer or y_buffer
        ax_right.set_ylim(y_min - pad, y_max + pad)
    else:
        ax_right.set_ylim(y_limit_right)

    # Styling
    ax_left.grid(False)
    ax_right.grid(False)
    for ax in [ax_left, ax_right]:
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)

    ax_left.set_aspect('auto', adjustable='datalim')
    fig.subplots_adjust(left=0.15, right=0.85, top=0.92, bottom=0.18)

    # Horizontal legend
    clean_label_left = ylabel_left.replace("\n", " ")
    clean_label_right = ylabel_right.replace("\n", " ")

    lines = [line_left, line_right]
    labels = [clean_label_left, clean_label_right]
    
    valid_locations = {
        'best': 0, 'upper right': 1, 'upper left': 2, 'lower left': 3,
        'lower right': 4, 'right': 5, 'center left': 6, 'center right': 7,
        'lower center': 8, 'upper center': 9, 'center': 10
    }
    
    if isinstance(legend_loc, str):
        loc = legend_loc.lower() if legend_loc.lower() in valid_locations else 'lower left'
    elif isinstance(legend_loc, int) and 0 <= legend_loc <= 10:
        loc = legend_loc
    else:
        loc = 'lower left'
    
    ax_left.legend(lines, labels, fontsize=14, frameon=False,
                   loc=loc, alignment='left', ncol=2)

    # Save or display
    if save_dir is not None:
        os.makedirs(os.path.dirname(save_dir) or '.', exist_ok=True)
        fig.savefig(
            save_dir, dpi=300, bbox_inches='tight',
            pad_inches=0.3, transparent=True
        )
        print(f"Saved plot to {save_dir}")
    else:
        plt.show()

    return


def line_multi_strategy(
    y_data_list,
    labels,
    xlabel="X", ylabel_left="Y",
    colors=None, linestyles=None,
    x_limit=None, y_limit=None,
    legend_loc='lower right', save_dir=None, y_buffer=0.4
):
    """
    Plot multiple line curves with a single y-axis on a wide rectangular figure.
    
    Parameters:
    -----------
    y_data_list : list of array-like
        List of data series to plot (each will be a separate line)
    labels : list of str
        Legend labels for each line
    xlabel, ylabel_left : str
        Axis labels
    colors : list of str or None
        Line colors. If None, uses default color cycle
    linestyles : list of str or None
        Line styles. If None, uses dashed lines for all
    x_limit, y_limit : tuple or None
        Axis limits (min, max)
    legend_loc : str or int
        Legend position ('upper left', 'lower right', 'best', etc.)
    save_dir : str or None
        Path to save figure
    y_buffer : float
        Y-axis padding factor (default 0.4 = 40%)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    # Default colors and linestyles
    if colors is None:
        colors = ['#6F8ABF', '#BF6F8A', '#8ABF6F', '#BF8A6F']
    if linestyles is None:
        linestyles = ['--'] * len(y_data_list)  # All dashed
    
    # Convert to numpy arrays
    y_data_list = [np.array(y, dtype=float) for y in y_data_list]
    
    # Ensure all have the same length
    lengths = [len(y) for y in y_data_list]
    if len(set(lengths)) > 1:
        raise ValueError("All y data series must have the same length.")
    
    # Auto-generate x values
    x = np.arange(lengths[0]) + 1
    
    # Wide rectangular figure
    fig, ax = plt.subplots(figsize=(12, 3.75))
    
    # Plot all curves with markers
    lines = []
    for i, y_data in enumerate(y_data_list):
        color = colors[i % len(colors)]
        linestyle = linestyles[i % len(linestyles)]
        label = labels[i] if i < len(labels) else f"Strategy {i+1}"
        
        line, = ax.plot(
            x, y_data, color=color, linestyle=linestyle,
            linewidth=2.5, marker='o', markersize=6, label=label
        )
        lines.append(line)
    
    # Axis labels
    ax.set_xlabel(xlabel, fontsize=20, color='black')
    ax.set_ylabel(ylabel_left, fontsize=20, color='black')
    
    # Tick label style
    ax.tick_params(axis='both', which='major', labelsize=14, colors='black')
    
    # Axis limits with padding
    if x_limit is not None:
        ax.set_xlim(x_limit)
    
    if y_limit is None:
        all_y = np.concatenate(y_data_list)
        y_min, y_max = np.min(all_y), np.max(all_y)
        pad = (y_max - y_min) * y_buffer or y_buffer
        ax.set_ylim(y_min - pad, y_max + pad)
    else:
        ax.set_ylim(y_limit)
    
    # Styling
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    
    ax.set_aspect('auto', adjustable='datalim')
    fig.subplots_adjust(left=0.15, right=0.85, top=0.92, bottom=0.18)
    
    # Vertical legend (one item per row)
    clean_labels = [label.replace("\n", " ") for label in labels[:len(y_data_list)]]
    
    valid_locations = {
        'best': 0, 'upper right': 1, 'upper left': 2, 'lower left': 3,
        'lower right': 4, 'right': 5, 'center left': 6, 'center right': 7,
        'lower center': 8, 'upper center': 9, 'center': 10
    }
    
    if isinstance(legend_loc, str):
        loc = legend_loc.lower() if legend_loc.lower() in valid_locations else 'lower right'
    elif isinstance(legend_loc, int) and 0 <= legend_loc <= 10:
        loc = legend_loc
    else:
        loc = 'lower right'
    
    # Changed ncol=1 to stack items vertically
    ax.legend(lines, clean_labels, fontsize=14, frameon=False,
              loc=loc, alignment='left', ncol=1)
    
    # Save or display
    if save_dir is not None:
        os.makedirs(os.path.dirname(save_dir) or '.', exist_ok=True)
        fig.savefig(
            save_dir, dpi=300, bbox_inches='tight',
            pad_inches=0.3, transparent=True
        )
        print(f"Saved plot to {save_dir}")
    else:
        plt.show()
    
    return

