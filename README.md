# Beas Cloudburst Watch

Interactive dashboard of machine-learning extreme-rainfall and cloudburst-type risk for the Beas basin (Kullu–Manali, Himachal Pradesh), JJAS monsoon 2001–2025.

- Maps: Leaflet on an OpenStreetMap basemap (map data © OpenStreetMap contributors, ODbL).
- Data: IMD 0.25° gridded rainfall, ERA5 reanalysis, NASA GPM IMERG, SRTM elevation; Getis-Ord Gi* hotspots computed with Apache Spark + Apache Sedona.
- Models: logistic regression, random forest, LightGBM and XGBoost, scored out of sample on 2019–2024 and 2025.

This is a research hindcast, not an operational warning system.

The whole dashboard is the single self-contained file `index.html`.
