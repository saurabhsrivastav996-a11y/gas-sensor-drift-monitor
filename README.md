# Gas Sensor Drift & Abnormal Trend Monitor

An educational, retrospective monitor built from the [UCI Air Quality dataset](https://archive.ics.uci.edu/dataset/360/air%2Bquality). It cleans timestamps, converts the source's -200 missing marker to nulls, compares indexed sensor and analyzer trends, and creates point-reading and sustained-divergence flags for investigation.

## Deliverables

- notebooks/gas_sensor_drift_monitor.ipynb — analysis, cleaning steps, thresholds, interpretation and proposed maintenance fields.
- PowerBI/GasSensorDriftDashboard.html — self-contained interactive preview with channel selection, indexed trends, weather context, missing rates and flagged periods.
- PowerBI/ — CSV tables ready to import into Power BI, DAX measures, field mapping and a report-page guide.
- data/raw/AirQualityUCI.csv — original UCI CSV, preserved for provenance.
- data/processed/air_quality_clean.csv — timestamp-cleaned source values; missing markers are blank.
- scripts/build_project.py — rebuilds the cleaned and Power BI tables, HTML preview and notebook from the included source CSV.

## Run the notebook

From this project folder, install the packages in requirements.txt, start Jupyter, then open notebooks/gas_sensor_drift_monitor.ipynb.

The companion script regenerates derived files from the source CSV with: python scripts/build_project.py

## Method and limits

Point flags use a trailing 24-hour median and MAD; absolute robust scores above 3.5 are flagged after at least 12 valid hours. Paired channels are also indexed independently to each channel's study-period median. A sustained divergence candidate uses a trailing 7-day rolling average of the index gap, an absolute robust score above 2.5, and at least 72 flagged hours in a 7-day period. The smoothed divergence rule uses a lower threshold than the point rule and requires at least three days of persistence.

These are screening thresholds, not device-failure diagnoses. Sensor response and reference-analyzer concentration have different units and behavior, so the dashboard compares direction with separate median-based indices. Weather and cross-sensitivity can affect readings. The source data also has substantial missingness, especially in NMHC(GT).

The downloaded CSV contains 9,357 rows with parseable timestamps. The UCI page describes 9,358 instances; blank timestamp rows in the distributed CSV are excluded. The source period in the parsed file is 10 March 2004 to 4 April 2005.

No calibration, asset-age, alarm, inspection or repair records are included. Recommended maintenance fields in the notebook and guide are proposed schema additions and are not presented as real records.

## Source and use

Vito, S. (2008). Air Quality [Dataset]. UCI Machine Learning Repository. [DOI: 10.24432/C59K5F](https://doi.org/10.24432/C59K5F). The UCI dataset page lists CC BY 4.0 and also states research use only, excluding commercial purposes; follow the source page's use terms and cite UCI.

## Power BI note

Power BI Desktop was not available in the build environment, so this project does not include a native .pbix file. The Power BI folder contains the import tables, measures and page instructions, plus the offline HTML preview for immediate interactive review.
