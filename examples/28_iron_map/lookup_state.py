import pandas as pd
import geopandas as gpd


# Load a US states shapefile (example)
# You can download this from the Census Bureau or use built-in datasets
states = gpd.read_file("tl_2024_us_state/tl_2024_us_state.shp")

# View the first few rows to see state codes (STUSPS) and names (NAME)
print(states.head())

# MN = states[states['STUSPS'] == 'MN']


# Example: Finding which state a specific long/lat is in
df = pd.read_csv("ned_reduced_sitelist.csv")
gdf_points = gpd.GeoDataFrame(
    df, geometry=gpd.points_from_xy(df["site.longitude"], df["site.latitude"])
)

# Set CRS to match the states shapefile (usually EPSG:4326 for lon/lat)
gdf_points.crs = "EPSG:4326"
states = states.to_crs("EPSG:4326")

# Spatial join points to states
joined = gpd.sjoin(gdf_points, states, how="left", predicate="intersects")
# joined.to_csv("ned_reduced_gdf.csv")

# Look up rates
df["state"] = None
df["rate"] = None
rates = pd.read_csv("rates.csv", skiprows=1, header=None, index_col=0)
for i in range(len(df)):
    index = df.index.values[i]
    point = joined.iloc[i]
    state = point["STUSPS"]
    rate = rates.loc[state].values[0]
    df.loc[index, "state"] = state
    df.loc[index, "rate"] = rate
df.to_csv("ned_reduced_gdf.csv")
