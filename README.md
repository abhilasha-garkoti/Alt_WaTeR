# Alt-WaTER

**Water level Time series Estimation for Reservoirs using satellite Altimetry**

---

## Description

Alt-WaTER is a *Python* software for generating reservoir water level time series from satellite radar altimetry data. The software combines multi-mission satellite altimetry, Global Surface Water (GSW) water masks, and Sentinel-1 images accessed through Google Earth Engine. User interaction is provided through a graphical user interface.

---

## Contributions

* **Abhilasha Garkoti**, Indian Institute of Technology Kanpur, India
* **Balaji Devaraju**, Indian Institute of Technology Kanpur, India

---

## Processing steps

For a given reservoir bounding box and a directory of altimetry NetCDF files, the software performs the following steps:

* Extraction of permanent water extent
* Extraction of maximum water extent
* Classification of the reservoir
* Filtering of altimetry measurements
* Computation of reservoir water levels
* Generation of output files and figures

---

## Graphical user interface

Alt-WaTER is intended to be used through a graphical user interface implemented using *streamlit*. The interface allows the user to:

* Enter reservoir bounding box coordinates
* Select a local directory containing altimetry NetCDF files
* Specify an output CSV file
* Authenticate Google Earth Engine
* Run the processing workflow
* View and download generated figures

---

## Supported satellite missions

| Mission                   | Radar mode   |
| ------------------------- | ------------ |
| Sentinel-3A / Sentinel-3B | SAR          |
| Sentinel-6A               | SAR HR / LRM |
| Jason-3                   | LRM          |
| SWOT (NALT)               | SAR          |

---

## Reservoir classification

Each reservoir is assigned to one of five classes based on altimetry track behaviour and surrounding terrain characteristics:

| Class | Description                                                   |
| ----: | ------------------------------------------------------------- |
|     1 | High-resolution altimetry fully over permanent water          |
|     2 | Low-resolution altimetry fully over permanent water           |
|     3 | High-resolution altimetry affected by shoreline contamination |
|     4 | Low-resolution altimetry affected by shoreline contamination  |
|     5 | Reservoirs in complex or mountainous terrain                  |

The assigned class controls buffering distance, filtering strategy, and the use of Sentinel-1 data.

---

## Water level computation

Water levels are computed using a median elevation approach:

1. Altimetry measurements within the water mask are selected
2. Elevation corrections are applied
3. The median elevation is calculated
4. The measurement closest to the median is retained

Outlier filtering is applied before generating the final time series.

---

## Software requirements

### Python

Python version **3.9 or later** is required.

### Required libraries

The software depends on the following Python libraries:

* *numpy*
* *pandas*
* *geopandas*
* *shapely*
* *scipy*
* *netCDF4*
* *matplotlib*
* *rioxarray*
* *pygmt*
* *earthengine-api*
* *geemap*
* *scikit-image*
* *streamlit*

Installation using **conda-forge** is recommended due to geospatial dependencies.

---

## Google Earth Engine authentication

Google Earth Engine is used for accessing Global Surface Water data and Sentinel-1 imagery. Authentication is required once per machine.

When using the graphical user interface, authentication is initiated through a browser-based login if credentials are not available. Credentials are stored locally after successful authentication.

---

## Running the graphical interface

From the project root directory:

```bash
streamlit run gui/alt_water.py
```

The interface opens in a web browser.

---

## Input data

### Reservoir extent

The reservoir area is defined using a geographic bounding box specified by minimum and maximum longitude and latitude values in decimal degrees (EPSG:4326).

### Altimetry data

The input directory must contain NetCDF files from supported satellite missions. Subdirectories are allowed.

### Output

The output CSV path must correspond to a writable location on the local system.

---

## Outputs

### CSV file

The output CSV file contains one record per satellite pass with the following fields:

* date
* mission
* cycle
* latitude
* longitude
* elevation
* uncertainty

### Figures

The interface displays and allows downloading of:

* Reservoir water level time series
* Reservoir map with water masks and altimetry points

---

## License

The software is distributed under the **GNU General Public License**. See the `LICENSE` file for details.

---

## Acknowledgements

The contributors would like to thank NOAA, USGS, Copernicus EU, JRC-GSW, GEE and GMT for providing access to the datasets used in this software. The development of this software benefited from open-source geospatial libraries including *numpy*, *pandas*, *geopandas*, *matplotlib*, *earthengine-api*, *geemap*, and *streamlit*.

---

## Directory structure

```text
Alt-WaTER/
├── core/
│   ├── reservoir_pipeline.py
│   ├── generate_time_series.py
│   ├── classification.py
│   ├── water_masks.py
│   └── altimetry_extractors.py
├── gui/
│   └── alt_water.py
├── README.md
└── LICENSE
```
