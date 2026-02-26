def generate_altimetry_timeseries(
    nc_folder,
    max_gdf,
    reservoir_class,
    output_dir,
    s1_search_days=90,
):
    """
    Generate a reservoir water-level time series using:

    Median elevation of all valid altimetry points per pass

    Filtering:
      - Global MAD filter (threshold = 6.0)
    """

    # --------------------------------------------------
    # Imports
    # --------------------------------------------------
    import os, glob
    import numpy as np
    import pandas as pd
    import geopandas as gpd
    import ee, geemap
    from datetime import datetime, timedelta, UTC
    from skimage.filters import threshold_otsu

    from .altimetry_extractors import extract_altimetry_data

    # --------------------------------------------------
    # Output path
    # --------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)
    MEDIAN_CSV = os.path.join(output_dir, "MED.csv")

    # --------------------------------------------------
    # CLASS PARSER
    # --------------------------------------------------
    def parse_reservoir_class(res_class):
        if not isinstance(res_class, str) or len(res_class) != 2:
            raise ValueError(f"Invalid reservoir_class: {res_class}")
        return int(res_class[0]), res_class[1]

    base_class, _ = parse_reservoir_class(reservoir_class)
    use_s1 = base_class in [3, 4]

    # −1 km inward buffer ONLY for Class 1–2
    buffer_m = -1000 if base_class in [1, 2] else 0

    # --------------------------------------------------
    # HELPERS
    # --------------------------------------------------
    def buffer_gdf(gdf, buffer_m):
        g = gdf.to_crs(3857)
        g["geometry"] = g.geometry.buffer(buffer_m)
        g = g[~g.is_empty]
        return g.to_crs(4326)

    def clean_sjoin_columns(gdf):
        drop_cols = [c for c in ["index_left", "index_right"] if c in gdf.columns]
        if drop_cols:
            gdf = gdf.drop(columns=drop_cols)
        return gdf

    def mad_filter(df, col, thr=6):
        """
        Standard 6MAD outlier removal

        Parameters
        ----------
        df : DataFrame (date-indexed OK)
        col : str
        thr : int (default = 6)

        Returns
        -------
        Filtered DataFrame
        """
        x = df[col]

        median = x.median()
        mad = np.median(np.abs(x - median))

        # Avoid divide-by-zero
        if mad == 0 or np.isnan(mad):
            return df.copy()

        z = 0.6745 * (x - median) / mad
        return df[np.abs(z) <= thr]

    
    # --------------------------------------------------
    # SENTINEL-1 MASK
    # --------------------------------------------------
    def sentinel1_mask(region_ee, target_date):
        start = ee.Date(target_date - timedelta(days=s1_search_days))
        end   = ee.Date(target_date + timedelta(days=s1_search_days))

        coll = (
            ee.ImageCollection("COPERNICUS/S1_GRD")
            .filterBounds(region_ee)
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
            .filterDate(start, end)
            .select("VV")
        )

        if coll.size().getInfo() == 0:
            return None, None

        target = ee.Date(target_date)

        def add_time_diff(img):
            diff = img.date().difference(target, "day").abs()
            return img.set("time_diff", diff)

        closest_img = ee.Image(coll.map(add_time_diff).sort("time_diff").first())
        closest_date = ee.Date(closest_img.get("system:time_start"))

        same_day = coll.filterDate(closest_date, closest_date.advance(1, "day"))
        mosaic = same_day.mosaic().clip(region_ee)
        vv = mosaic.select("VV")

        # Otsu threshold
        sample = vv.sample(region=region_ee, scale=30, numPixels=5000, geometries=False)
        arr = np.array(sample.aggregate_array("VV").getInfo())
        arr = arr[np.isfinite(arr)]

        if arr.size < 100:
            return None, None

        thr = threshold_otsu(arr)
        water_mask = vv.lt(thr).rename("water")

        s1_date = closest_date.format("YYYY-MM-dd").getInfo()
        return water_mask, s1_date


    buffered_max = buffer_gdf(max_gdf, buffer_m)
    max_union = buffered_max.geometry.union_all()
    region_ee = geemap.geopandas_to_ee(buffered_max)

    # --------------------------------------------------
    # OUTPUT CONTAINER
    # --------------------------------------------------
    ts_median = []

    # --------------------------------------------------
    # MAIN LOOP
    # --------------------------------------------------
    for nc in sorted(glob.glob(os.path.join(nc_folder, "**/*.nc"), recursive=True)):
        try:
            df = extract_altimetry_data(nc)
        except Exception:
            continue

        required = {"latitude", "longitude", "date", "mission", "altitude", "range"}
        if not required.issubset(df.columns):
            continue

        # Fill missing corrections
        for c in ["iono", "dry", "wet", "pole", "solid","load"]:
            if c not in df.columns:
                df[c] = 0.0
            df[c] = df[c].fillna(0.0)

        if "geoid" not in df.columns:
            df["geoid"] = 0.0
        df["geoid"] = df["geoid"].fillna(0.0)

        corr= df["dry"] + df["wet"] + df["pole"] + df["solid"] + df["iono"]+df["load"]

        #elevation
        df["elevation"] = df["altitude"]- (df["range"] + corr) - df["geoid"]
        

        df = df.dropna(subset=["elevation", "latitude", "longitude", "date","cycle"])

        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df.longitude, df.latitude),
            crs="EPSG:4326",
        )

        gdf = gdf[gdf.geometry.within(max_union)]
        if gdf.empty:
            continue

        # Sentinel-1 water mask (Class 3–4 only)
        if use_s1:

            # Get Sentinel-1 mask
            mask, s1_date = sentinel1_mask(
                region_ee,
                pd.to_datetime(gdf["date"].iloc[0])
            )

            # No SAR data → skip this file
            if mask is None:
                continue

            # Convert points to EE
            fc = geemap.geopandas_to_ee(gdf)

            # ---- SAFE water fraction extractor ----
            def add_water_fraction(feat):
                geom = feat.geometry()

                # Compute mean water fraction inside 100 m buffer
                result = mask.reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=geom.buffer(100),  # YOUR BUFFER
                    scale=10,
                    bestEffort=True,
                    maxPixels=1e6
                )

                # Force property existence even if NULL
                frac = ee.Algorithms.If(
                    result.contains("water"),
                    result.get("water"),
                    -1  # marker for missing
                )

                return feat.set("water_frac", frac)

            # Map over features
            fc = fc.map(add_water_fraction)

            # Convert back to GeoDataFrame
            gdf = geemap.ee_to_gdf(fc)

            # --------------------------------------------------
            # SAFETY CHECKS (prevents KeyError)
            # --------------------------------------------------

            # If column completely missing → skip safely
            if "water_frac" not in gdf.columns:
                continue

            # Convert to numeric safely
            gdf["water_frac"] = pd.to_numeric(gdf["water_frac"], errors="coerce")

            # Remove EE missing marker (-1)
            gdf.loc[gdf["water_frac"] < 0, "water_frac"] = np.nan

            # Drop rows where SAR had no info
            gdf = gdf.dropna(subset=["water_frac"])
            if gdf.empty:
                continue

            # --------------------------------------------------
                    # Water classification
            # --------------------------------------------------
            gdf["water_flag"] = (gdf["water_frac"] >= 0.4).astype(int)
            gdf["s1_date"] = s1_date

            # Keep only water points
            gdf = gdf[gdf["water_flag"] == 1]
            if gdf.empty:
                continue


        gdf = clean_sjoin_columns(gdf)

        mission = gdf["mission"].iloc[0]
        date = pd.to_datetime(gdf["date"].iloc[0])

        # ---------------- Median time series ----------------
        med = gdf["elevation"].median()

        # Step 2: Find observed point closest to median
        idx = (gdf["elevation"] - med).abs().idxmin()
        pt = gdf.loc[idx]

        # Step 3: Representative observed elevation (raw measurement)
        e_rep = pt["elevation"]

        # Step 4: Standard deviation relative to representative elevation
        std_rep = np.std(gdf["elevation"] - e_rep, ddof=1)

        ts_median.append({
            "mission": mission,
            "date": date,
            "cycle":pt["cycle"],

            # ✅ Use representative elevation, not the median value
            "elevation": e_rep,

            # Representative coordinates
            "latitude": pt["latitude"],
            "longitude": pt["longitude"],

            "uncertainty": std_rep,
        })

    # --------------------------------------------------
    # SAVE OUTPUT WITH FILTERING
    # --------------------------------------------------
    if ts_median:
        ts = (
            pd.DataFrame(ts_median)
            .drop_duplicates("date")
            .sort_values("date")
            .set_index("date")
        )

        ts = mad_filter(ts, "elevation")
        
        ts.reset_index().to_csv(
            MEDIAN_CSV, index=False, float_format="%.3f"
        )

    return MEDIAN_CSV
