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
    local_crs="EPSG:2263", shift=True, city_shp_path=None, offset=0.0015
):
    import geopandas as gpd
    import networkx as nx

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
    assert mapbox_token is not None, 'Missing mapbox mb_token.'
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


def map_roads_plotly(geo_roads, mapbox_token=None, save_dir='',):
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


def map_roads_w_values(
        geo_roads, values, local_crs="EPSG:2263",
        shift=True, city_shp_path=None, offset=0.0015
):
    import geopandas as gpd
    fig, ax = plt.subplots(figsize=(10, 10))

    # base map
    if city_shp_path is not None:
        geo_city = gpd.read_file(city_shp_path).to_crs(local_crs)
        geo_city.plot(ax=ax, facecolor='lightgray', edgecolor='none')

    # roads
    roads = geo_roads.copy()
    if shift:
        roads['direction'] = roads['geometry'].apply(_road_geo_get_direction)
        roads['geometry'] = roads.apply(
            lambda row: _road_geo_directional_shift(row['geometry'], row['direction'], offset=offset),
            axis=1
        )
    roads["value"] = roads['link_id'].map(values)
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


def dist_gmm_1d(gmm, speed_range=None, title=None):
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

    plt.title(title)
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


def dist_gmm_3d(gmm, segment1=None, segment2=None, speed_range=(0, 90), z_limit=None, return_z_max=False):
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

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(
        X, Y, Z,
        cmap=_truncate_colormap('viridis', 0.15, 1),
        edgecolor='none', antialiased=False, alpha=1
    )

    ax.set_xlabel(f'Speed on Link {segment1} (mph)', fontsize=14, labelpad=12)
    ax.set_ylabel(f'Speed on Link {segment2} (mph)', fontsize=14, labelpad=12)
    ax.set_zlabel('Density', fontsize=14, labelpad=12)  # labels
    ax.ticklabel_format(axis='z', style='sci', scilimits=(-2, 2))
    ax.zaxis.get_offset_text().set_fontsize(12)  # scientific notation

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

    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.tick_params(axis='z', labelsize=14)  # font size

    xs = [ax.get_xlim()[0], ax.get_xlim()[0]]
    ys = [ax.get_ylim()[0], ax.get_ylim()[0]]
    zs = [ax.get_zlim()[0], ax.get_zlim()[1]]
    ax.plot(xs, ys, zs, color='black', linewidth=1)  # verticle line on the left

    xticks = np.linspace(x.min(), x.max(), 7)
    yticks = np.linspace(y.min(), y.max(), 7)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)  # spase label

    plt.show()

    if return_z_max:
        return ax.get_zlim()[1]


def dist_discrete_gmm(dist_w_obs, gmm_wo_abs=None, close_curve=[]):
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

    fig, ax = plt.subplots(figsize=(7.7, 7))
    plt.rc('font', size=28)
    if gmm_wo_abs is not None:
        ax.plot(speed_range, pdf_wo_abs, label=f'w/o observation', color='grey', linestyle='--')
        # ax.fill_between(speed_range, pdf_wo_abs, alpha=0.2, color='grey')
    if 'no_flood' not in close_curve:
        ax.plot(speed_range, pdf_no_flood, label=f'No Flood (p={p_no_flood:.3f})', color='#0339A6')
        ax.fill_between(speed_range, pdf_no_flood, alpha=0.2, color='#4174D9')
    if 'flood' not in close_curve:
        ax.plot(speed_range, pdf_flood, label=f'Flood (p={p_flood:.3f})', color='#D91604')
        ax.fill_between(speed_range, pdf_flood, alpha=0.2, color='#D95448')
    ax.set_xlabel('Speed (mph)')
    ax.set_ylabel('Density')
    ax.legend()

    ax.yaxis.set_major_formatter(mtick.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))  # scientific notation

    ax.legend(frameon=False, fontsize=16)

    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    plt.tight_layout(pad=.5)
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
        xscale='log', yscale='log', reg='linear', base=10,):
    import numpy as np
    import os
    from scipy import stats

    if xscale not in ('linear', 'log',):
        raise ValueError("xscale must be one of 'linear', or 'log'")
    if yscale not in ('linear', 'log',):
        raise ValueError("xscale must be one of 'linear', or 'log'")
    if reg not in ('robust', 'linear',):
        raise ValueError("xscale must be one of 'robust', or 'linear'")
    
    x = np.asarray(kld)
    y = np.asarray(kld_estimated)

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
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(x, y, alpha=0.85, edgecolor='k', s=60, marker='o')
    ax.set_xlabel('KLD 1', fontsize=12)
    ax.set_ylabel('KLD 2', fontsize=12)

    ax.plot(xs, y_line, color='red', linestyle='--', linewidth=2,
            label='Trend (fitted on transformed x)')
    
    if xscale == 'log':
        ax.set_xscale('log', base=base)
    if yscale == 'log':
        ax.set_yscale('log', base=base)

    ax.legend()

    r2_text = f'$R^2$ = {r2:.3f}' if np.isfinite(r2) else '$R^2$ = N/A'
    ax.annotate(r2_text, xy=(0.02, 0.95), xycoords='axes fraction',
                ha='left', va='top', fontsize=11,
                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))

    plt.tight_layout()
    if out_file:
        os.makedirs(os.path.dirname(out_file) or '.', exist_ok=True)
        fig.savefig(out_file, dpi=300)
        print(f'Saved scatter plot to {out_file}')
    plt.show()
    return



