import os
import numpy as np
import pandas as pd
import geopandas as gpd
import pygmt
import rioxarray

from scipy.ndimage import generic_filter
from .altimetry_extractors import extract_altimetry_data, discover_altimetry_files

pygmt.config(GMT_COMPATIBILITY="6")

# ==========================================================
# RESOLUTION CHECK
# ==========================================================
def is_high_resolution(filename: str) -> bool:
    f = filename.upper()
    return (
        "ENHANCED_MEASUREMENT" in f or
        "SWOT" in f or
        ("S6" in f and "__HR_" in f)
    )

# ==========================================================
# VECTOR LOADER
# ==========================================================
def load_vector(input_obj):
    gdf = input_obj.copy() if isinstance(input_obj, gpd.GeoDataFrame) else gpd.read_file(input_obj)
    return gdf.to_crs("EPSG:4326")

# ==========================================================
# INWARD BUFFER (−1 km, MAX EXTENT)
# ==========================================================
def inward_buffer_km(gdf, buffer_km):
    lon, _ = gdf.geometry.union_all().centroid.xy
    utm_zone = int((lon[0] + 180) / 6) + 1
    utm_crs = f"EPSG:{32600 + utm_zone}"

    gdf_utm = gdf.to_crs(utm_crs)
    buffered = gdf_utm.buffer(-buffer_km * 1000)
    buffered = buffered[~buffered.is_empty]

    return gpd.GeoDataFrame(
        geometry=buffered,
        crs=utm_crs
    ).to_crs("EPSG:4326")

# ==========================================================
# TERRAIN HELPERS
# ==========================================================
def horn_slope_deg(z, dx, dy):
    zp = np.pad(z, 1, constant_values=np.nan)
    dzdx = ((zp[:-2,2:] + 2*zp[1:-1,2:] + zp[2:,2:]) -
            (zp[:-2,:-2] + 2*zp[1:-1,:-2] + zp[2:,:-2])) / (8 * dx)
    dzdy = ((zp[2:,:-2] + 2*zp[2:,1:-1] + zp[2:,2:]) -
            (zp[:-2,:-2] + 2*zp[:-2,1:-1] + zp[:-2,2:])) / (8 * dy)
    return np.degrees(np.arctan(np.sqrt(dzdx**2 + dzdy**2)))

def riley_tri(z):
    def tri(w):
        c = w[4]
        n = np.delete(w, 4)
        n = n[np.isfinite(n)]
        return np.sqrt(np.sum((c - n)**2)) if np.isfinite(c) and len(n) >= 5 else np.nan
    return generic_filter(z, tri, size=3, mode="constant", cval=np.nan)

# ==========================================================
# BUFFER FROM ALL TRACKS
# ==========================================================
def buffer_from_all_tracks(points_inside_water, max_water_gdf, buffer_km):
    hull = points_inside_water.unary_union.convex_hull
    lon, _ = hull.centroid.xy
    utm_zone = int((lon[0] + 180) / 6) + 1
    utm_crs = f"EPSG:{32600 + utm_zone}"

    hull_utm = gpd.GeoDataFrame(
        geometry=[hull],
        crs="EPSG:4326"
    ).to_crs(utm_crs)

    water_utm = max_water_gdf.to_crs(utm_crs)

    buffer_land = hull_utm.buffer(buffer_km * 1000).difference(
        water_utm.union_all()
    )
    buffer_land = buffer_land.buffer(0)

    return gpd.GeoDataFrame(
        geometry=buffer_land,
        crs=utm_crs
    ).to_crs("EPSG:4326")

# ==========================================================
# TERRAIN + GEOID METRICS
# ==========================================================
def compute_terrain_indices_from_buffer(buffer_geom):
    geom = buffer_geom.to_crs("EPSG:4326")
    bounds = geom.total_bounds

    region = [
        bounds[0] - 0.1,
        bounds[2] + 0.1,
        bounds[1] - 0.1,
        bounds[3] + 0.1,
    ]

    dem = (
        pygmt.datasets.load_earth_relief("03s", region=region)
        .rio.write_crs("EPSG:4326")
        .load()
    )

    dem_clip = dem.rio.clip(geom.geometry, geom.crs)
    z = dem_clip.values.astype(float)

    tri = riley_tri(z)

    return {
        "elev_p95_p05_m": float(np.nanpercentile(z, 95) - np.nanpercentile(z, 5)),
        "tri_p95_m": float(np.nanpercentile(tri, 95)),
    }


# ==========================================================
# TERRAIN COMPLEXITY (AT LEAST 2 FAILURES)  ← ONLY CHANGE
# ==========================================================
def is_complex_terrain(base_folder, max_water_input):
    max_water = load_vector(max_water_input)
    all_points = []

    for f in discover_altimetry_files(base_folder):
        df = extract_altimetry_data(f)
        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df.longitude, df.latitude),
            crs="EPSG:4326"
        )
        all_points.append(gdf)

    pts_all = gpd.GeoDataFrame(
        pd.concat(all_points, ignore_index=True),
        crs="EPSG:4326"
    )

    inside = gpd.sjoin(
        pts_all,
        max_water,
        predicate="intersects",
        how="inner"
    )

    if inside.empty:
        return False, pd.DataFrame()

    buffer_land = buffer_from_all_tracks(inside, max_water, 5)
    metrics = compute_terrain_indices_from_buffer(buffer_land)

    failures = sum([
        metrics["elev_p95_p05_m"] >= 500,
        metrics["tri_p95_m"] >= 100,
    
    ])

    complex_flag = failures >= 1

    return complex_flag, pd.DataFrame([metrics])

# ==========================================================
# TRACK GEOMETRY CHECK (1/2 vs 3/4)
# ==========================================================
def check_altimetry_tracks(base_folder, max_water_input, perm_water_input):
    max_water = load_vector(max_water_input)
    perm_water = load_vector(perm_water_input)
    max_inner = inward_buffer_km(max_water, 1)

    records = []

    for f in discover_altimetry_files(base_folder):
        try:
            df = extract_altimetry_data(f)
            if df.empty:
                continue

            gdf = gpd.GeoDataFrame(
                df,
                geometry=gpd.points_from_xy(df.longitude, df.latitude),
                crs="EPSG:4326"
            )

            max_hits = gpd.sjoin(gdf, max_water, predicate="intersects", how="inner")
            if max_hits.empty:
                continue

            perm_hits = gpd.sjoin(
                max_hits.drop(columns="index_right", errors="ignore"),
                perm_water,
                predicate="intersects",
                how="inner",
            )

            inner_hits = gpd.sjoin(
                perm_hits.drop(columns="index_right", errors="ignore"),
                max_inner,
                predicate="intersects",
                how="inner",
            )

            records.append({
                "file": os.path.basename(f),
                "passes": (not perm_hits.empty) and (not inner_hits.empty)
            })

        except Exception:
            continue

    return pd.DataFrame(records)

# ==========================================================
# MASTER DRIVER
# ==========================================================
def run_full_classification(base_folder, max_water_input, perm_water_input):

    has_hr = any(
        is_high_resolution(os.path.basename(f))
        for f in discover_altimetry_files(base_folder)
    )

    track_results = check_altimetry_tracks(
        base_folder,
        max_water_input,
        perm_water_input
    )

    base_class = (
        1 if has_hr else 2
    ) if track_results["passes"].all() else (
        3 if has_hr else 4
    )

    complex_flag, terrain_metrics = is_complex_terrain(
        base_folder,
        max_water_input
    )

    return f"{base_class}{'B' if complex_flag else 'A'}", track_results, terrain_metrics
