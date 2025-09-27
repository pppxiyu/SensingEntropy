import pandas as pd
import geopandas as gpd
import visualization as vis


class RoadData:
    def __init__(self):
        self.geo = None  # geometry of roads
        self.speed = None
        self.speed_resampled = None
        self.closures = None
        self.flood_time_citywide = None
        self.flood_time_per_road = None

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
                'and save it in ./cache/unreadable_nyc_roads.csv'
            )
            unreadable = pd.read_csv('cache/unreadable_nyc_roads.csv', dtype=str, delimiter=';', index_col=None)
            if string.encode('unicode_escape').decode() in unreadable['encoded_polyline'].values:
                points = unreadable.loc[
                    unreadable['encoded_polyline'] == string.encode('unicode_escape').decode(), 'points'
                ].values[0]
                points = re.sub(r'\)\(', '), (', points)
                points = ast.literal_eval(points)
                print('Read from ./cache/unreadable_nyc_roads.csv')
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
                # self.geo = data
                return
            else:
                print("Road geometry file does not exist. Pull it.")
        else:
            if os.path.exists(f'./cache/speed/nyc_traffic_{time_range[0]}_{time_range[1]}.csv'):
                print("Traffic file exists")
                data = pd.read_csv(
                    f'./cache/speed/nyc_traffic_{time_range[0]}_{time_range[1]}.csv',
                    dtype={"link_id": str}, parse_dates=["time"]
                )
                data = self._remove_segment_w_many_na(data)
                self.speed = data
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
            # self.geo = all_data
        else:
            all_data = all_data.rename(columns={'DATA_AS_OF': 'time', 'LINK_ID': 'link_id', 'SPEED': 'speed'})
            time_range[0] = time_range[0].replace(':', '-').replace('T', '-')
            time_range[1] = time_range[1].replace(':', '-').replace('T', '-')
            all_data.to_csv(f'./cache/speed/nyc_traffic_{time_range[0]}_{time_range[1]}.csv', index=False)
            self.speed = all_data
        return all_data

    def import_street_flooding(self, dir_file, crs, buffer):
        from shapely.wkt import loads

        df = pd.read_csv(dir_file,  dtype={'Due Date': str})
        df = df[~df['Location'].isna()]
        df.loc[:, 'Location'] = df['Location'].apply(loads)

        gdf = df.rename(columns={
            'Location': 'geometry', 'Unique Key': 'id', 'Created Date': 'time_create', 'Closed Date': 'time_close',
            'Street Name': 'st_name', 'Cross Street 1': 'cross_st_1', 'Cross Street 2': 'cross_st_2',
        })
        gdf['time_create'] = pd.to_datetime(gdf['time_create'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
        gdf['time_close'] = pd.to_datetime(gdf['time_close'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
        gdf = gpd.GeoDataFrame(gdf, geometry='geometry').set_crs('4326')
        gdf = gdf[['id', 'geometry', 'time_create', ]]

        # filter closure using the buffer zones of roads
        road_geo = self.geo.to_crs(crs).copy()
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
        closure_refine['id'] = closure_refine['id'].astype(str)
        self.closures = closure_refine
        return closure_refine

    def resample_nyc_dot_traffic(self, data, time_interval='5min', save_2_class=True):
        data.loc[:, 'time'] = data['time'].dt.round(time_interval)
        data.set_index("time", inplace=True)
        data_resampled = data.groupby(['link_id', 'time'])["speed"].mean().reset_index()
        if save_2_class:
            self.speed_resampled = data_resampled
        return data_resampled

    @staticmethod
    def _remove_all_zero_segment(data, threshold=.75):
        data_pivot = data.pivot(index="time", columns="link_id", values="speed")
        zero_percent = (data_pivot == 0).sum() / (~data_pivot.isna()).sum()
        valid_segments = zero_percent[zero_percent < threshold].index
        data = data[data['link_id'].isin(valid_segments)]
        return data

    @staticmethod
    def _remove_segment_w_many_na(data, time_interval='5min', threshold=.5):
        max_count = data['time'].dt.round(time_interval).nunique()
        value_per = (data.groupby('link_id').count() / max_count)['speed']
        valid_seg = value_per[value_per > threshold]
        data = data[data['link_id'].isin(valid_seg.index.to_list())]
        return data

    def filter_road_closure(self, crs, buffer):
        from shapely.ops import unary_union
        road_geo = self.closures.to_crs(crs)
        road_closure = self.closures.to_crs(crs)
        road_closure_filtered = road_closure[road_closure.geometry.within(unary_union(road_geo.buffer(buffer).geometry))]
        road_closure_filtered = road_closure_filtered.to_crs(epsg=4326)
        self.closures = road_closure_filtered
        return

    def import_adapted_nyc_road(self, dir_roads):
        gdf = gpd.read_file(dir_roads)
        gdf = gdf[['link_id', 'link_name', 'geometry']]
        self.geo = gdf
        return

    @staticmethod
    def infer_flooding_time(df, pre_flood_buffer, post_flood_buffer):
        df['buffer_start'] = df['time_create'] - pd.Timedelta(hours=pre_flood_buffer)
        df['buffer_end'] = df['time_create'] + pd.Timedelta(hours=post_flood_buffer)

        df = df.sort_values('time_create').reset_index(drop=True)
        non_overlap_df = pd.DataFrame(columns=['buffer_start', 'buffer_end']).astype(
            {'buffer_start': 'datetime64[ns]', 'buffer_end': 'datetime64[ns]'}
        )
        previous_end = None
        for index, row in df.iterrows():
            start = row['buffer_start']
            end = row['buffer_end']
            if previous_end is None or start >= previous_end:
                non_overlap_df = pd.concat(
                    [non_overlap_df, pd.DataFrame({'buffer_start': [start], 'buffer_end': [end]})],
                    ignore_index=True
                )
                previous_end = end
            elif end > previous_end:
                adjusted_start = previous_end
                non_overlap_df = pd.concat(
                    [non_overlap_df, pd.DataFrame({'buffer_start': [adjusted_start], 'buffer_end': [end]})],
                    ignore_index=True)
                previous_end = end

        non_overlap_df = non_overlap_df.sort_values('buffer_start').reset_index(drop=True)
        combined_df = pd.DataFrame(columns=['buffer_start', 'buffer_end']).astype(
            {'buffer_start': 'datetime64[ns]', 'buffer_end': 'datetime64[ns]'}
        )
        current_start = non_overlap_df.iloc[0]['buffer_start']
        current_end = non_overlap_df.iloc[0]['buffer_end']
        for i in range(1, len(non_overlap_df)):
            next_start = non_overlap_df.iloc[i]['buffer_start']
            next_end = non_overlap_df.iloc[i]['buffer_end']
            if next_start == current_end:
                current_end = next_end
            else:
                combined_df = pd.concat(
                    [combined_df, pd.DataFrame({'buffer_start': [current_start], 'buffer_end': [current_end]})],
                    ignore_index=True
                )
                current_start = next_start
                current_end = next_end
        combined_df = pd.concat(
            [combined_df, pd.DataFrame({'buffer_start': [current_start], 'buffer_end': [current_end]})],
            ignore_index=True
        )
        return combined_df

    def infer_flooding_time_citywide(self, pre_flood_buffer, post_flood_buffer):
        self.flood_time_citywide = self.infer_flooding_time(self.closures.copy(), pre_flood_buffer, post_flood_buffer)
        return

    def infer_flooding_time_per_road(self, pre_flood_buffer, post_flood_buffer):
        flood_time_per_road = {}
        closures = self.closures.copy()
        for road in closures['link_id'].unique():
            closures_select = closures.loc[closures['link_id'] == road, :]
            flood_time_per_road[road] = self.infer_flooding_time(closures_select.copy(), pre_flood_buffer, post_flood_buffer)
        self.flood_time_per_road = flood_time_per_road
        return

    def pull_nyc_dot_traffic_flooding(self, nyc_data_token, select_incidents=None, verbose=0):
        import os
        data_list = []

        if select_incidents is None:
            select_df = self.flood_time_citywide.copy()
        else:
            select_df = select_incidents.copy()

        for _, row in select_df.iterrows():
            start = row['buffer_start']
            end = row['buffer_end']

            if os.path.exists(
                    f"./cache/speed/nyc_traffic_"
                    f"{start.strftime('%Y-%m-%dT%H:%M:%S').replace(':', '-').replace('T', '-')}_"
                    f"{end.strftime('%Y-%m-%dT%H:%M:%S').replace(':', '-').replace('T', '-')}.csv"
            ):
                print("Traffic file exists")
            else:
                print("Traffic file does not exist. Pull it.")
                self.pull_nyc_dot_traffic(
                    nyc_data_token,
                    [start.strftime('%Y-%m-%dT%H:%M:%S'), end.strftime('%Y-%m-%dT%H:%M:%S')],
                    False
                )

            try:
                data = pd.read_csv(
                    f"./cache/speed/nyc_traffic_"
                    f"{start.strftime('%Y-%m-%dT%H:%M:%S').replace(':', '-').replace('T', '-')}_"
                    f"{end.strftime('%Y-%m-%dT%H:%M:%S').replace(':', '-').replace('T', '-')}.csv",
                    dtype={"link_id": str}, parse_dates=["time"]
                )
                data_list.append(data)
            except pd.errors.EmptyDataError:
                pass

        if len(data_list) > 0:
            df_concat = pd.concat(data_list, ignore_index=True)
            df_concat = self._remove_segment_w_many_na(df_concat)
            self.speed = df_concat
        else:
            if verbose > 0:
                import warnings
                warnings.warn('No data is extracted.')
            return None
        return self.speed

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
            print('Road Data Class exists')
            return True
        except Exception as e:
            print('Failed to load Road Data, build from scratch')
            return False

    def get_assigned_time_interval(self, road_name):
        flood_time_citywide = self.flood_time_citywide.copy()
        flood_time_citywide['buffer_start'] = pd.to_datetime(flood_time_citywide['buffer_start'])
        flood_time_citywide['buffer_end'] = pd.to_datetime(flood_time_citywide['buffer_end'])
        df = self.flood_time_per_road[road_name].copy()
        df['buffer_start'] = pd.to_datetime(df['buffer_start'])
        df['buffer_end'] = pd.to_datetime(df['buffer_end'])

        df['buffer_start_citywide'] = pd.NaT
        df['buffer_end_citywide'] = pd.NaT

        for idx, row in df.iterrows():
            match = flood_time_citywide[
                (flood_time_citywide['buffer_start'] <= row['buffer_start']) &
                (flood_time_citywide['buffer_end'] >= row['buffer_end'])
                ]
            assert len(match) == 1, "Multiple matches found or no match found"

            df.at[idx, 'buffer_start_citywide'] = match.iloc[0]['buffer_start']
            df.at[idx, 'buffer_end_citywide'] = match.iloc[0]['buffer_end']

        return df

    def get_data_when_sensing_on(self, road_name):
        df = self.get_assigned_time_interval(road_name)
        import os
        data_list = []
        for _, row in df.iterrows():
            start = row['buffer_start_citywide']
            end = row['buffer_end_citywide']

            assert os.path.exists(
                    f"./cache/speed/nyc_traffic_"
                    f"{start.strftime('%Y-%m-%dT%H:%M:%S').replace(':', '-').replace('T', '-')}_"
                    f"{end.strftime('%Y-%m-%dT%H:%M:%S').replace(':', '-').replace('T', '-')}.csv"
            ), 'Data not exist'

            try:
                data = pd.read_csv(
                    f"./cache/speed/nyc_traffic_"
                    f"{start.strftime('%Y-%m-%dT%H:%M:%S').replace(':', '-').replace('T', '-')}_"
                    f"{end.strftime('%Y-%m-%dT%H:%M:%S').replace(':', '-').replace('T', '-')}.csv",
                    dtype={"link_id": str}, parse_dates=["time"]
                )
                data_list.append(data)
            except pd.errors.EmptyDataError:
                pass

        df_concat = pd.concat(data_list, ignore_index=True)
        df_concat = self._remove_segment_w_many_na(df_concat)
        df_concat = self.resample_nyc_dot_traffic(df_concat, save_2_class=False)
        return df_concat

    @staticmethod
    def get_flooded_roads_during_inct(flood_time_per_road: dict, inct: pd.DataFrame, overlap_threshold=0.5):
        assert len(inct) == 1, 'Only one incident is considered here.'

        incident_start = pd.to_datetime(inct.iloc[0]['buffer_start'])
        incident_end = pd.to_datetime(inct.iloc[0]['buffer_end'])
        incident_duration = (incident_end - incident_start).total_seconds()

        flooded_segments = []
        for segment_id, df in flood_time_per_road.items():
            for _, row in df.iterrows():
                flood_start = pd.to_datetime(row['buffer_start'])
                flood_end = pd.to_datetime(row['buffer_end'])

                overlap_start = max(incident_start, flood_start)
                overlap_end = min(incident_end, flood_end)
                if overlap_start < overlap_end:
                    overlap_duration = (overlap_end - overlap_start).total_seconds()
                    overlap_rate = overlap_duration / incident_duration
                    if overlap_rate >= overlap_threshold:
                        flooded_segments.append(segment_id)
                        break  # One match is enough
        return flooded_segments
    
    @ staticmethod
    def calculate_link_speed_correlation(df):
        data_pivot = df.pivot(index="time", columns="link_id", values="speed")
        correlation_matrix = data_pivot.corr()
        return correlation_matrix


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


def get_period_before_start(ict: pd.DataFrame, pre_period_days=7):

    pre_incident = ict.copy()
    pre_incident['buffer_end'] = pre_incident['buffer_start']
    pre_incident['buffer_start'] = pre_incident['buffer_start'] - pd.Timedelta(days=pre_period_days)

    return pre_incident


if __name__ == '__main__':
    pass

