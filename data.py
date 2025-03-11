import pandas as pd
import geopandas as gpd
import visualization as vis


class RoadData:
    def __init__(self):
        self.road_geo = None
        self.road_speed = None
        self.road_closure = None
        self.road_closure_p = None
        self.road_speed_resampled = None

    @staticmethod
    def _parse_encoded_lines(string):
        from shapely.geometry import LineString
        import polyline
        import ast
        import re
        try:
            decoded_line = polyline.decode(string)
            if any([(abs(i[0]) < 1 or abs(i[1]) < 10) for i in decoded_line]):
                decoded_line = [i for i in decoded_line if not (abs(i[0]) < 1 or abs(i[1]) < 10)]
            if len(decoded_line) >= 2:
                return LineString([(i[1], i[0]) for i in decoded_line])
            else:
                return None
        except Exception as e:
            print(
                'Unreadable roads by polyline package. '
                'Manually check it at https://developers.google.com/maps/documentation/utilities/polylineutility'
                'and save it in ./data/unreadable_NYC_roads.csv'
            )
            unreadable = pd.read_csv('./data/unreadable_NYC_roads.csv', dtype=str, delimiter=';', index_col=None)
            if string.encode('unicode_escape').decode() in unreadable['encoded_polyline'].values:
                points = unreadable.loc[
                    unreadable['encoded_polyline'] == string.encode('unicode_escape').decode(), 'points'
                ].values[0]
                points = re.sub(r'\)\(', '), (', points)
                points = ast.literal_eval(points)
                print('Read from ./data/unreadable_NYC_roads.csv')
                return LineString([(i[1], i[0]) for i in points])
            return None

    def pull_nyc_dot_traffic(self, dir_token, time_range, geo_only, visual_check=True):
        import requests
        import time
        import os

        url = "https://data.cityofnewyork.us/resource/i4gi-tjb9.json"
        with open(dir_token, "r") as file:
            api_token = file.read().strip()
        selected_columns = "SPEED,DATA_AS_OF,LINK_ID"

        if geo_only:
            if os.path.exists(f'./cache/nyc_roads_geo.geojson'):
                print("Road geometry file exists")
                data = gpd.read_file(f'./cache/nyc_roads_geo.geojson')
                # self.road_geo = data
                return
            else:
                print("Road geometry file does not exist. Pull it.")
        else:
            if os.path.exists(f'./cache/nyc_traffic_{time_range[0]}_{time_range[1]}.csv'):
                print("Traffic file exists")
                data = pd.read_csv(
                    f'./cache/nyc_traffic_{time_range[0]}_{time_range[1]}.csv',
                    dtype={"link_id": str}, parse_dates=["time"]
                )
                data = self._remove_all_zero_segment(data)
                self.road_speed = data
                return
            else:
                print("Traffic file does not exist. Pull it.")

        headers = {
            "x-App-Token": api_token
        }
        batch_size = 50000
        offset = 0
        all_data = []
        while True:
            params = {
                "$select": selected_columns,
                "$where": f"DATA_AS_OF >= '{time_range[0]}' AND DATA_AS_OF <= '{time_range[1]}'",
                "$limit": batch_size,
                "$offset": offset,
                "$order": "DATA_AS_OF ASC"
            }
            if geo_only:
                params = {
                    "$select": 'LINK_ID,ENCODED_POLY_LINE,LINK_NAME',
                    "$where": f"DATA_AS_OF >= '{time_range[0]}' AND DATA_AS_OF <= '{time_range[1]}'",
                    "$limit": batch_size,
                    "$offset": offset,
                    "$order": "DATA_AS_OF ASC"
                }
            response = requests.get(url, headers=headers, params=params)
            if response.status_code == 200:
                data = response.json()
                num_records = len(data)
                print(f"Retrieved {num_records} records at offset {offset}")
                all_data.extend(data)
                if num_records < batch_size:
                    break
                offset += batch_size
                time.sleep(.5)
            else:
                print(f"Error: {response.status_code} - {response.text}")
                break

        print(f"Total records retrieved: {len(all_data)}")
        for record in all_data[:5]:
            print(record)

        all_data = pd.DataFrame(all_data)
        if geo_only:
            all_data = all_data.loc[
                all_data.groupby('LINK_ID')['ENCODED_POLY_LINE'].apply(lambda x: x.str.len().idxmin())
            ]
            all_data.loc[:, 'LINK_LINE'] = all_data['ENCODED_POLY_LINE'].apply(self._parse_encoded_lines)
            all_data = all_data.rename(columns={'LINK_LINE': 'geometry', 'LINK_ID': 'link_id', 'LINK_NAME': 'link_name'})
            all_data = all_data[~all_data['geometry'].isna()]
            all_data = all_data[['link_id', 'geometry', 'link_name']]
            all_data = gpd.GeoDataFrame(all_data, geometry='geometry').set_crs(epsg=4326)

            if visual_check:
                import matplotlib.pyplot as plt
                all_data.plot()
                plt.show()

            all_data.to_file(f'./cache/nyc_roads_geo.geojson', driver="GeoJSON")
            # self.road_geo = all_data
        else:
            all_data = all_data.rename(columns={'DATA_AS_OF': 'time', 'LINK_ID': 'link_id', 'SPEED': 'speed'})
            all_data.to_csv(f'./cache/nyc_traffic_{time_range[0]}_{time_range[1]}.csv', index=False)
            self.road_speed = all_data
        return all_data

    def import_street_flooding(self, dir_file, crs, buffer):
        from shapely.wkt import loads

        df = pd.read_csv(dir_file,  dtype={'Due Date': str})
        df = df[~df['Location'].isna()]
        df['Location'] = df['Location'].apply(loads)

        gdf = df.rename(columns={
            'Location': 'geometry', 'Unique Key': 'id', 'Created Date': 'time_create', 'Closed Date': 'time_close',
            'Street Name': 'st_name', 'Cross Street 1': 'cross_st_1', 'Cross Street 2': 'cross_st_2',
        })
        gdf['time_create'] = pd.to_datetime(gdf['time_create'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
        gdf['time_close'] = pd.to_datetime(gdf['time_close'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
        gdf = gpd.GeoDataFrame(gdf, geometry='geometry').set_crs('4326')
        gdf = gdf[['id', 'geometry', 'time_create',]]

        road_geo = self.road_geo.to_crs(crs).copy()
        road_geo_buffer = road_geo.copy()
        road_geo_buffer.geometry = road_geo_buffer.buffer(buffer)
        road_closure = gdf.to_crs(crs)
        closure_refine = road_closure.sjoin(road_geo_buffer, how='inner')
        closure_refine['distance'] = closure_refine.apply(
            lambda row: road_geo[road_geo['link_id'] == row['link_id']].distance(row.geometry).values[0], axis=1
        )
        closure_refine = closure_refine.reset_index(drop=True)
        closure_refine = closure_refine.loc[
            closure_refine.groupby('id')['distance'].idxmin()
        ]
        closure_refine = closure_refine[['id', 'geometry', 'time_create', 'link_id']]
        closure_refine = closure_refine.to_crs(epsg=4326)
        self.road_closure = closure_refine
        return closure_refine

    def resample_nyc_dot_traffic(self, data):
        data.set_index("time", inplace=True)
        data_resampled = data.groupby("link_id").resample("5min")["speed"].mean().reset_index()
        self.road_speed_resampled = data_resampled
        return data_resampled

    @staticmethod
    def _remove_all_zero_segment(data, threshold=0.75):
        data_pivot = data.pivot(index="time", columns="link_id", values="speed")
        zero_percent = (data_pivot == 0).sum() / (~data_pivot.isna()).sum()
        valid_segments = zero_percent[zero_percent < threshold].index
        data = data[data['link_id'].isin(valid_segments)]
        return data

    def filter_road_closure(self, crs, buffer):
        from shapely.ops import unary_union
        road_geo = self.road_closure.to_crs(crs)
        road_closure = self.road_closure.to_crs(crs)
        road_closure_filtered = road_closure[road_closure.geometry.within(unary_union(road_geo.buffer(buffer).geometry))]
        road_closure_filtered = road_closure_filtered.to_crs(epsg=4326)
        self.road_closure = road_closure_filtered
        return

    def convert_closure_2_prob(self, interval='1D', if_vis=False):
        if if_vis:
            vis.bar_closure_count_over_time(self.road_closure.copy())
        pass
        closures = self.road_closure.copy()
        closures = closures.set_index('time_create')
        closure_count = closures.groupby('link_id').count()['id']
        daily_counts = closures.resample(interval).size().fillna(0)
        flooding_day_count = len(daily_counts[daily_counts > 0])
        closure_count_p = closure_count / flooding_day_count
        closure_count_p = closure_count_p.reset_index().rename(columns={'id': 'p'})
        self.road_closure_p = closure_count_p
        return

    def import_adapted_nyc_road(self, dir_roads):
        gdf = gpd.read_file(dir_roads)
        gdf = gdf[['link_id', 'link_name', 'geometry']]
        self.road_geo = gdf
        return


def _polyline_parse(string):
    from shapely.geometry import LineString
    from shapely.geometry import Point, box
    try:
        points = string.split()
        cleaned_points = []
        for point in points:
            try:
                bbox_geom = box(40.47, -74.26, 40.95, -73.68)
                xy = point.split(',')
                point_geom = Point(xy[0], xy[1])
                assert point_geom.within(bbox_geom)
                cleaned_points.append((float(xy[1]), float(xy[0])))
            except Exception as e:
                pass
        return LineString(cleaned_points)
    except Exception as e:
        return None


def import_nyc_dot_traffic_legacy(dir_file):
    # used for data manually downloaded from NYC Data Portal
    df = pd.read_csv(dir_file)
    df_link_geo = df.drop_duplicates(subset=['LINK_ID']).copy()
    df_link_geo.loc[:, 'LINK_LINE'] = df_link_geo['LINK_POINTS'].apply(_polyline_parse)
    df_merge_geo = df.merge(df_link_geo[['LINK_ID', 'LINK_LINE']], how='left', on='LINK_ID')

    gdf = df_merge_geo.rename(columns={
        'LINK_LINE': 'geometry',
        'ID': 'id', 'SPEED': 'speed', 'TRAVEL_TIME': 'travel_time',
        'DATA_AS_OF': 'time', 'LINK_ID': 'link_id',
    })
    gdf = gpd.GeoDataFrame(gdf, geometry='geometry').set_crs(epsg=4326)
    return gdf


def merge_road_speed_geo(road_speed, road_geo):
    road_speed = road_speed.merge(road_geo, on='link_id', how='inner')
    road_speed = gpd.GeoDataFrame(road_speed, geometry='geometry').set_crs(epsg=4326)
    return road_speed




