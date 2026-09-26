# Power BI dashboard build guide

## Import tables

In Power BI Desktop, use Get data → Text/CSV to import these files:

1. SensorMonitoring.csv — hourly fact table (one row per timestamp and sensor channel).
2. ChannelQuality.csv — channel-level missingness and flag counts.
3. InvestigationEvents.csv — contiguous periods for review.
4. FieldReference.csv — sensor target and comparison notes.

Confirm Timestamp, StartTime, and EndTime are Date/Time; values and rates are decimal numbers; the flag and missing columns are True/False. Keep empty values as blanks. If the import wizard infers a flag as text, change it to Boolean in Power Query.

## One-page layout

- Top bar: title and a visible note: “Flags are investigation items, not proof of sensor failure.”
- Slicers: SensorChannel and Timestamp date range.
- Cards: Sensor Missing Rate, Sensor Point Flags, Sustained Divergence Candidate Hours, Mean Temperature C, and Mean Relative Humidity Pct. The DAX measures are in Measures.dax.
- Trend visual: Date/Time on the x-axis; Sensor Trend Index and Reference Trend Index as lines. Filter to a single sensor channel. Both lines are independently normalized to their own study-period median (=100), so this visual compares direction only.
- Weather visual: temperature and relative humidity as separate visuals or small multiples, with their units shown.
- Investigation table: sensor, reference, flag type, start/end, flagged hours, peak robust score, mean temperature and mean relative humidity from InvestigationEvents.
- Missingness table: channel and missing rate from ChannelQuality.

Use a date hierarchy only if it preserves the user's ability to inspect the time period. For a dense hourly display, aggregate to daily median indices and daily weather averages.

## Interpretation notes

- Sensor channels and analyzer channels do not share units. Do not compare their raw values on a shared axis.
- The 24-hour robust point threshold is 3.5. The sustained divergence threshold is 2.5 on a 7-day rolling-average index gap, with at least 72 flagged hours in the 7-day window.
- These thresholds produce investigation candidates. They do not establish failure or maintenance need.
- The UCI data has no calibration, sensor age, alarm, inspection or repair history. Connect those as a future maintenance table keyed by a stable asset identifier; do not fill them with simulated records.

## Dashboard preview

Open GasSensorDriftDashboard.html in a browser for a self-contained interactive preview of the proposed page. Power BI Desktop is required to save a native PBIX report.
