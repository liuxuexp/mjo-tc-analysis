import os
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, LineString
from shapely.ops import nearest_points
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Union
from shapely.ops import unary_union
import warnings
warnings.filterwarnings('ignore')

class CMATyphoonProcessor:
    def __init__(self, data_path: str, china_shapefile: str):
        """
        Initialize CMA typhoon data processor
        """
        self.data_path = Path(data_path)
        self.china_shapefile = china_shapefile
        self.china_polygon = None
        self.china_boundary = None
        self.load_china_boundary()

        self.all_typhoons_data = []
        self.landfall_typhoons_data = []
        self.non_landfall_typhoons_data = []
        self.landfall_info = []

        self.stats = {
            'total_typhoons': 0,
            'processed_typhoons': 0,
            'skipped_typhoons': 0,
            'landfall_typhoons': 0,
            'years_processed': []
        }

    def load_china_boundary(self):

        print("Loading China boundary data...")
        china_gdf = gpd.read_file(self.china_shapefile)

        if china_gdf.crs is None or china_gdf.crs.to_epsg() != 4326:
            china_gdf = china_gdf.to_crs(epsg=4326)

        ref_points = {
            'mainland': Point(116.4, 39.9),
            'taiwan':   Point(121.5, 25.0),
            'hainan':   Point(110.3, 20.0)
        }

        selected_polygons = []

        for name, pt in ref_points.items():
            found = False
            for idx, row in china_gdf.iterrows():
                geom = row.geometry
                if geom.contains(pt):
                    if geom.geom_type == 'MultiPolygon':
                        for subpoly in geom.geoms:
                            if subpoly.contains(pt):
                                selected_polygons.append(subpoly)
                                found = True
                                break
                    else:
                        selected_polygons.append(geom)
                        found = True
                    break
            if not found:
                print(f"  Warning: no polygon containing reference point {name} found, will try area sorting.")

        if len(selected_polygons) < 3:
            print("  Insufficient polygons from reference points, enabling area sorting fallback...")
            china_gdf['area'] = china_gdf.geometry.area
            china_gdf = china_gdf.sort_values('area', ascending=False)
            top3 = china_gdf.iloc[:3].geometry
            for geom in top3:
                if not any(geom.equals(ex) for ex in selected_polygons):
                    selected_polygons.append(geom)

        if selected_polygons:
            self.china_polygon = unary_union(selected_polygons)
            self.china_boundary = self.china_polygon.boundary
            print(f"  Loaded {len(selected_polygons)} polygons (mainland, Taiwan, Hainan)")
        else:
            raise ValueError("Could not extract any valid land polygons from shapefile.")

    def is_in_china_mainland(self, lat: float, lon: float) -> bool:
        if pd.isna(lat) or pd.isna(lon):
            return False
        point = Point(lon, lat)
        return self.china_polygon.buffer(1e-9).contains(point)

    def get_wind_category(self, wind_speed: float) -> int:

        if wind_speed is None or pd.isna(wind_speed):
            return 0
        if wind_speed <= 17.1:
            return 0
        elif wind_speed <= 32.6:
            return 1
        elif wind_speed < 51.0:
            return 2
        else:
            return 3

    def read_typhoon_data_fixed(self, typhoon_txt: Path, code: Union[str, int]) -> Tuple[Dict, pd.DataFrame]:
        typhoon_txt = Path(typhoon_txt)
        if isinstance(code, int):
            code = "{:04}".format(code)

        with open(typhoon_txt, "r") as txt_handle:
            while True:
                header = txt_handle.readline().split()
                if not header:
                    raise ValueError(f"Typhoon code {code} not found in file")
                if header[4].strip() == code:
                    break
                [txt_handle.readline() for _ in range(int(header[2]))]

            data_path = pd.read_table(
                txt_handle,
                sep=r"\s+",
                header=None,
                names=["TIME", "I", "LAT", "LONG", "PRES", "WND", "OWD"],
                nrows=int(header[2]),
                dtype={"I": int, "LAT": float, "LONG": float, "PRES": float,
                       "WND": float, "OWD": float},
                parse_dates=True,
                date_parser=lambda x: pd.to_datetime(x, format="%Y%m%d%H"),
                index_col="TIME",
            )

            data_path["LAT"] = data_path["LAT"] / 10
            data_path["LONG"] = data_path["LONG"] / 10
            data_path.loc[data_path["WND"] == 9, "WND"] = 9.5
            data_path.loc[data_path["WND"] == 0, "WND"] = np.nan

            name_parts = header[7:-1]
            typhoon_name = ' '.join(name_parts) if name_parts else ''

            header_info = {
                'classification': header[0],
                'international_code': header[1],
                'num_records': header[2],
                'sequence_number': header[3],
                'chinese_code': header[4],
                'end_flag': header[5],
                'interval': header[6],
                'name': typhoon_name
            }
            return header_info, data_path

    def _compute_landfall_point(self, prev_row: pd.Series, curr_row: pd.Series) -> Tuple[float, float]:
        """
        Compute precise intersection point between typhoon track segment and China land boundary.
        Returns (lat, lon), or current point coordinates if intersection fails.
        """
        prev_lon, prev_lat = prev_row['LONG'], prev_row['LAT']
        curr_lon, curr_lat = curr_row['LONG'], curr_row['LAT']

        line = LineString([(prev_lon, prev_lat), (curr_lon, curr_lat)])

        intersection = line.intersection(self.china_boundary)

        if intersection.is_empty:
            return curr_lat, curr_lon

        if intersection.geom_type == 'Point':
            lon, lat = intersection.x, intersection.y
        elif intersection.geom_type == 'MultiPoint':
            points = list(intersection.geoms)
            nearest = min(points, key=lambda p: p.distance(Point(prev_lon, prev_lat)))
            lon, lat = nearest.x, nearest.y
        elif intersection.geom_type == 'LineString':
            coords = list(intersection.coords)
            nearest = min(coords, key=lambda c: Point(c).distance(Point(curr_lon, curr_lat)))
            lon, lat = nearest
        elif intersection.geom_type == 'GeometryCollection':
            points = [g for g in intersection.geoms if g.geom_type == 'Point']
            if points:
                nearest = min(points, key=lambda p: p.distance(Point(prev_lon, prev_lat)))
                lon, lat = nearest.x, nearest.y
            else:
                lon, lat = curr_lon, curr_lat
        else:
            lon, lat = curr_lon, curr_lat

        return lat, lon

    def detect_landfall(self, data: pd.DataFrame) -> List[Dict]:
        """
        Detect typhoon landfall events using precise geometric intersection for landfall point.
        """
        landfall_events = []
        prev_in_land = False
        prev_row = None

        for idx, curr_row in data.iterrows():
            lat, lon = curr_row['LAT'], curr_row['LONG']
            in_land = self.is_in_china_mainland(lat, lon)

            if prev_row is None:
                prev_in_land = in_land
                prev_row = curr_row
                continue

            if in_land and not prev_in_land:
                landfall_lat, landfall_lon = self._compute_landfall_point(prev_row, curr_row)

                landfall_wind = curr_row['WND']
                landfall_wind_category = self.get_wind_category(landfall_wind)

                landfall_events.append({
                    'time': idx,
                    'lat': landfall_lat,
                    'lon': landfall_lon,
                    'wind_speed': landfall_wind,
                    'pressure': curr_row['PRES'],
                    'wind_category': landfall_wind_category
                })

            prev_in_land = in_land
            prev_row = curr_row

        return landfall_events

    def process_typhoon(self, file_path: Path, typhoon_code: str, year: int,
                       print_sample: bool = False) -> Optional[Dict]:
        try:
            header, data = self.read_typhoon_data_fixed(file_path, typhoon_code)
        except Exception as e:
            print(f"  Failed to read typhoon {typhoon_code}: {e}")
            return None

        chinese_code = header['chinese_code']
        typhoon_name = header['name']

        if print_sample and len(data) > 0:
            print(f"\n  Typhoon {chinese_code} ({typhoon_name}) sample data (first 3 rows):")
            print(data.head(3))

        max_wind = data['WND'].max() if not data['WND'].isna().all() else 0
        max_wind_category = self.get_wind_category(max_wind)

        skip_reason = None
        if chinese_code == '0000':
            skip_reason = "Chinese code is 0000"
        elif max_wind_category == 0:
            skip_reason = f"Max wind category is 0 (max wind: {max_wind:.1f} m/s)"

        if skip_reason:
            print(f"  Skipping typhoon {chinese_code} ({typhoon_name}): {skip_reason}")
            data_copy = data.copy()
            data_copy['chinese_code'] = chinese_code
            data_copy['name'] = typhoon_name
            data_copy['max_wind'] = max_wind
            data_copy['wind_category'] = max_wind_category
            data_copy['year'] = year
            self.all_typhoons_data.append(data_copy)
            self.stats['skipped_typhoons'] += 1
            self.stats['total_typhoons'] += 1
            return None

        landfall_events = self.detect_landfall(data)

        typhoon_info = {
            'year': year,
            'chinese_code': chinese_code,
            'name': typhoon_name,
            'max_wind': max_wind,
            'max_wind_category': max_wind_category,
            'num_records': len(data),
            'has_landfall': len(landfall_events) > 0,
            'landfall_events': landfall_events
        }

        data_copy = data.copy()
        data_copy['chinese_code'] = chinese_code
        data_copy['name'] = typhoon_name
        data_copy['max_wind'] = max_wind
        data_copy['wind_category'] = max_wind_category
        data_copy['year'] = year

        self.all_typhoons_data.append(data_copy)
        self.stats['processed_typhoons'] += 1
        self.stats['total_typhoons'] += 1

        if landfall_events:
            landfall_events.sort(key=lambda x: x['wind_speed'] if not pd.isna(x['wind_speed']) else 0,
                               reverse=True)
            main_landfall = landfall_events[0]

            landfall_record = {
                'year': year,
                'chinese_code': chinese_code,
                'name': typhoon_name,
                'landfall_time': main_landfall['time'],
                'landfall_lat': main_landfall['lat'],
                'landfall_lon': main_landfall['lon'],
                'landfall_wind_speed': main_landfall['wind_speed'],
                'landfall_pressure': main_landfall['pressure'],
                'landfall_wind_category': main_landfall['wind_category'],
                'max_wind': max_wind,
                'max_wind_category': max_wind_category
            }
            self.landfall_info.append(landfall_record)
            self.landfall_typhoons_data.append(data_copy)
            self.stats['landfall_typhoons'] += 1
            print(f"  Typhoon {chinese_code} landfall detected, precise landfall point: "
                  f"({main_landfall['lat']:.4f}, {main_landfall['lon']:.4f}), "
                  f"landfall wind speed: {main_landfall['wind_speed']:.1f} m/s")
        else:
            self.non_landfall_typhoons_data.append(data_copy)

        return typhoon_info

    def get_all_typhoon_codes(self, file_path: Path) -> List[str]:
        """Get all typhoon codes from the file"""
        typhoon_codes = []
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line:
                i += 1
                continue
            parts = line.split()
            if len(parts) >= 5 and parts[0] == '66666':
                chinese_code = parts[4]
                if chinese_code not in typhoon_codes:
                    typhoon_codes.append(chinese_code)
                try:
                    num_records = int(parts[2])
                    i += 1 + num_records
                except:
                    i += 1
            else:
                i += 1
        return typhoon_codes

    def process_year(self, year: int, print_sample: bool = True) -> Dict:
        """Process typhoon data for one year"""
        print(f"\nProcessing year {year}...")
        file_path = self.data_path / f"CH{year}BST.txt"
        if not file_path.exists():
            print(f"  File not found: {file_path}")
            return None

        typhoon_codes = self.get_all_typhoon_codes(file_path)
        print(f"  Found {len(typhoon_codes)} typhoons")

        if print_sample:
            print(f"\n  First 3 lines of {year} file:")
            with open(file_path, 'r', encoding='utf-8') as f:
                for _ in range(3):
                    line = f.readline().strip()
                    if line:
                        print(f"    {line}")

        year_stats = {
            'year': year,
            'total': len(typhoon_codes),
            'processed': 0,
            'skipped': 0,
            'landfall': 0
        }

        for i, code in enumerate(typhoon_codes, 1):
            if i % 10 == 0 or i == 1 or i == len(typhoon_codes):
                print(f"  Processing typhoon {i}/{len(typhoon_codes)}: {code}")
            print_sample_data = (i <= 3) and print_sample
            result = self.process_typhoon(file_path, code, year, print_sample_data)
            if result is None:
                year_stats['skipped'] += 1
            else:
                year_stats['processed'] += 1
                if result['has_landfall']:
                    year_stats['landfall'] += 1

        print(f"  {year} complete: total {year_stats['total']}, "
              f"processed {year_stats['processed']}, skipped {year_stats['skipped']}, "
              f"landfall {year_stats['landfall']}")
        self.stats['years_processed'].append(year)
        return year_stats

    def save_results(self, output_dir: str = './typhoon_output'):
        """Save processing results"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nSaving results to {output_dir}...")

        if self.landfall_info:
            landfall_df = pd.DataFrame(self.landfall_info)
            columns_order = [
                'year', 'chinese_code', 'name', 'landfall_time',
                'landfall_lat', 'landfall_lon', 'landfall_wind_speed',
                'landfall_pressure', 'landfall_wind_category',
                'max_wind', 'max_wind_category'
            ]
            existing_columns = [col for col in columns_order if col in landfall_df.columns]
            landfall_df = landfall_df[existing_columns]
            landfall_file = output_dir / 'landfall_typhoons_info.csv'
            landfall_df.to_csv(landfall_file, index=False, encoding='utf-8-sig')
            print(f"  Landfall typhoon info saved to: {landfall_file} ({len(landfall_df)} records)")
            print(f"  First 3 records:")
            print(landfall_df.head(3).to_string())

        if self.all_typhoons_data:
            all_tracks_df = pd.concat(self.all_typhoons_data, ignore_index=False)
            all_tracks_file = output_dir / 'all_typhoons_tracks.csv'
            cols = ['chinese_code', 'name', 'year', 'max_wind', 'wind_category',
                   'LAT', 'LONG', 'PRES', 'WND', 'I', 'OWD']
            existing_cols = [col for col in cols if col in all_tracks_df.columns]
            all_tracks_df = all_tracks_df[existing_cols]
            all_tracks_df.to_csv(all_tracks_file, encoding='utf-8-sig')
            print(f"  All typhoon track data saved to: {all_tracks_file} "
                  f"({len(self.all_typhoons_data)} typhoons, {len(all_tracks_df)} records)")

        if self.landfall_typhoons_data:
            landfall_tracks_df = pd.concat(self.landfall_typhoons_data, ignore_index=False)
            landfall_tracks_file = output_dir / 'landfall_typhoons_tracks.csv'
            cols = ['chinese_code', 'name', 'year', 'max_wind', 'wind_category',
                   'LAT', 'LONG', 'PRES', 'WND', 'I', 'OWD']
            existing_cols = [col for col in cols if col in landfall_tracks_df.columns]
            landfall_tracks_df = landfall_tracks_df[existing_cols]
            landfall_tracks_df.to_csv(landfall_tracks_file, encoding='utf-8-sig')
            print(f"  Landfall typhoon track data saved to: {landfall_tracks_file} "
                  f"({len(self.landfall_typhoons_data)} typhoons)")

        if self.non_landfall_typhoons_data:
            non_landfall_tracks_df = pd.concat(self.non_landfall_typhoons_data, ignore_index=False)
            non_landfall_tracks_file = output_dir / 'non_landfall_typhoons_tracks.csv'
            cols = ['chinese_code', 'name', 'year', 'max_wind', 'wind_category',
                   'LAT', 'LONG', 'PRES', 'WND', 'I', 'OWD']
            existing_cols = [col for col in cols if col in non_landfall_tracks_df.columns]
            non_landfall_tracks_df = non_landfall_tracks_df[existing_cols]
            non_landfall_tracks_df.to_csv(non_landfall_tracks_file, encoding='utf-8-sig')
            print(f"  Non-landfall typhoon track data saved to: {non_landfall_tracks_file} "
                  f"({len(self.non_landfall_typhoons_data)} typhoons)")

        stats_file = output_dir / 'processing_statistics.txt'
        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write("CMA Tropical Cyclone Processing Statistics\n")
            f.write("=" * 50 + "\n")
            f.write(f"Years processed: {min(self.stats['years_processed'])}-{max(self.stats['years_processed'])}\n")
            f.write(f"Total typhoons: {self.stats['total_typhoons']}\n")
            f.write(f"Processed typhoons: {self.stats['processed_typhoons']}\n")
            f.write(f"Skipped typhoons: {self.stats['skipped_typhoons']}\n")
            f.write(f"Landfall typhoons: {len(self.landfall_info)}\n")
            f.write(f"Non-landfall typhoons: {len(self.non_landfall_typhoons_data)}\n")
        print(f"  Processing statistics saved to: {stats_file}")

    def validate_results(self):
        """Validate consistency of processing results"""
        print("\n" + "="*60)
        print("Validating processing results...")
        print("="*60)

        total_typhoons = len(self.all_typhoons_data)
        landfall_typhoons = len(self.landfall_typhoons_data)
        non_landfall_typhoons = len(self.non_landfall_typhoons_data)

        print(f"Total typhoons: {total_typhoons}")
        print(f"Landfall typhoons: {landfall_typhoons}")
        print(f"Non-landfall typhoons: {non_landfall_typhoons}")
        print(f"Landfall info records: {len(self.landfall_info)}")

        issues = []
        expected_total = landfall_typhoons + non_landfall_typhoons + self.stats['skipped_typhoons']
        if total_typhoons == expected_total:
            print("OK: typhoon count validation passed")
        else:
            issues.append(f"Typhoon count mismatch: total({total_typhoons}) != landfall({landfall_typhoons}) + non_landfall({non_landfall_typhoons}) + skipped({self.stats['skipped_typhoons']})")
            print(f"FAIL: {issues[-1]}")

        if landfall_typhoons == len(self.landfall_info):
            print("OK: landfall typhoon count validation passed")
        else:
            issues.append(f"Landfall typhoon count mismatch: track data({landfall_typhoons}) != info records({len(self.landfall_info)})")
            print(f"FAIL: {issues[-1]}")

        if self.stats['total_typhoons'] == total_typhoons:
            print("OK: statistics validation passed")
        else:
            issues.append(f"Statistics mismatch: recorded({self.stats['total_typhoons']}) != actual({total_typhoons})")
            print(f"FAIL: {issues[-1]}")

        if not issues:
            print("\nAll validations passed!")
        else:
            print(f"\nFound {len(issues)} issues:")
            for i, issue in enumerate(issues, 1):
                print(f"  {i}. {issue}")

        return {
            'total_typhoons': total_typhoons,
            'landfall_typhoons': landfall_typhoons,
            'non_landfall_typhoons': non_landfall_typhoons,
            'issues': issues
        }

    def process_all_years(self, start_year: int = 1960, end_year: int = 2024):


        all_stats = []
        for year in range(start_year, end_year + 1):
            stats = self.process_year(year, print_sample=(year in [1960, 1970, 1980, 1990, 2000, 2010, 2020]))
            if stats:
                all_stats.append(stats)

        self.print_summary_statistics(all_stats)
        self.save_results()
        self.validate_results()

    def print_summary_statistics(self, all_stats: List[Dict]):
        """Print summary statistics"""
        print("\n" + "="*60)
        print("Processing Summary Statistics")
        print("="*60)

        total_typhoons = sum(s['total'] for s in all_stats)
        total_processed = sum(s['processed'] for s in all_stats)
        total_skipped = sum(s['skipped'] for s in all_stats)
        total_landfall = sum(s['landfall'] for s in all_stats)

        print(f"Total typhoons: {total_typhoons}")
        print(f"Total processed: {total_processed}")
        print(f"Total skipped: {total_skipped}")
        print(f"Total landfall: {total_landfall}")

        if total_processed > 0:
            print(f"Landfall ratio: {total_landfall/total_processed*100:.1f}%")
            print(f"Skip ratio: {total_skipped/total_typhoons*100:.1f}%")

        decades = {}
        for year in range(1960, 2025, 10):
            decade_start = year
            decade_end = min(year + 9, 2024)
            decade_key = f"{decade_start}s"
            decade_stats = [s for s in all_stats if decade_start <= s.get('year', 0) <= decade_end]
            if decade_stats:
                decade_total = sum(s.get('total', 0) for s in decade_stats)
                decade_landfall = sum(s.get('landfall', 0) for s in decade_stats)
                decade_processed = sum(s.get('processed', 0) for s in decade_stats)
                landfall_ratio = decade_landfall/decade_processed*100 if decade_processed > 0 else 0
                decades[decade_key] = {'total': decade_total, 'landfall': decade_landfall, 'ratio': landfall_ratio}

        print("\nBy decade:")
        for decade, stats in decades.items():
            print(f"  {decade}: {stats['total']} typhoons, "
                  f"{stats['landfall']} landfall ({stats['ratio']:.1f}%)")


def main():
    """Main function"""
    data_dir = "./CMABSTdata"
    china_shapefile = "./shapefiles/china_country.shp"

    processor = CMATyphoonProcessor(data_dir, china_shapefile)
    try:
        processor.process_all_years(start_year=1960, end_year=2024)
        print("\n" + "="*60)
        print("Processing complete!")
        print("="*60)
    except Exception as e:
        print(f"\nError during processing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
