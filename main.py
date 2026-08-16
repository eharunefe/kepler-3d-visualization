import pandas as pd
import numpy as np

df = pd.read_csv("cumulative.csv", comment="#")

print(f"Total Rows: {df.shape[0]} \nTotal Columns: {df.shape[1]}")

print("\n---Types---")
print(df["koi_disposition"].value_counts())

candidates = df[df["koi_disposition"] == "CANDIDATE"].copy()

net_df = df[df["koi_disposition"] != "CANDIDATE"].copy()
net_df_p = df[df["koi_disposition"] == "CONFIRMED"].copy()

net_df["target"] = net_df["koi_disposition"].apply(
    lambda x: 1 if x == "CONFIRMED" else 0
)


ignore_cols = [
    "rowid",
    "kepid",
    "kepoi_name",
    "kepler_name",
    "koi_score",
    "koi_disposition",
    "koi_pdisposition",
    "koi_tce_delivname",
    "target"
]

X = net_df.drop(
    columns=[c for c in ignore_cols if c in net_df.columns]
)
Y = net_df["target"]


X = X.select_dtypes(include=[np.number])
X = X.fillna(X.median())


#post-preparation part
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=200, random_state=8, n_jobs=-1)
model.fit(X, Y)


print("\n---The Most Important Features---")
featureImportances = pd.Series(
    model.feature_importances_, index=X.columns
).sort_values(ascending=False)
print(featureImportances.head(5))


X_can = candidates.drop(
    columns=[c for c in ignore_cols if c in candidates.columns]
)
X_can = X_can.select_dtypes(include=[np.number])
X_can = X_can.fillna(X.median())

X_can = X_can[X.columns]
candidates["PlanetProbability"] = model.predict_proba(X_can)[:,1]





top_can = candidates.sort_values(
    by="PlanetProbability", ascending=False
)
print("\n---The Most Possible Planets---")
display_cols = [c for c in ["kepoi_name", "PlanetProbability", "koi_period", "koi_prad", "koi_teq"] if c in top_can.columns]

print(top_can[display_cols].head(10).to_string(index=False))



#Let's check the planets whether habitable or not
isRocky = top_can["koi_prad"].between(0.6, 1.6) #Rocky Surface
isBright = top_can["koi_insol"].between(0.35, 1.11) #Stellar Flux
isMainSequence = (top_can["koi_slogg"] > 4.0) #Non-Giant Star
isStableStar = top_can["koi_steff"].between(4000, 6540) #Stable Star
isRelativelyHighConfidence = (top_can["PlanetProbability"] > 0.2)

habitableOnesV = top_can[isRocky & isBright & isMainSequence & isStableStar & isRelativelyHighConfidence]

#control
cols = [
    "kepoi_name",
    "PlanetProbability",
    "koi_prad",
    "koi_insol",
    "koi_period",
    "koi_steff",
    "koi_slogg"
]
available_cols = [c for c in cols if c in habitableOnesV.columns]

print("\n---Habitable Zone Candidates---")
print(
    habitableOnesV[available_cols].to_string(index=False)
)


#Let's check the certain planets
isRocky = net_df_p["koi_prad"].between(0.6, 1.6) #Rocky Surface
isBright = net_df_p["koi_insol"].between(0.35, 1.11) #Stellar Flux
isMainSequence = (net_df_p["koi_slogg"] > 4.0) #Non-Giant Star
isStableStar = net_df_p["koi_steff"].between(4000, 6540) #Stable Star

habitableOnes = net_df_p[isRocky & isBright & isMainSequence & isStableStar]

#control
cols = [
    "kepler_name",
    "kepoi_name",
    "koi_prad",
    "koi_insol",
    "koi_period",
    "koi_steff",
    "koi_slogg"
]
available_cols = [c for c in cols if c in habitableOnes.columns]

print("\n---Habitable Zone At Confirmed Planets---")
print(
    habitableOnes[available_cols].to_string(index=False)
)




#_________________Stage 2 >> Visualization
df = df.copy()
#radius
df["ra"] = np.radians(df["ra"])
df["dec"] = np.radians(df["dec"])


#distance
T_SUN = 5778.0
M_SUN = 4.74

#Stellar Luminosity (Stefan-Boltzmann: L ~ R^2 * T^4)
df["lum"] = (df["koi_srad"] ** 2) * ((df["koi_steff"] / T_SUN) ** 4)

#Absolute Magnitude
lumClean = df["lum"].clip(lower=1e-5)
df["M_REAL"] = M_SUN - 2.5 * np.log10(lumClean)

#Distance (parsec_distance = 10 ^ ((m - M + 5) / 5))
df["distance"] = (10 ** ((df["koi_kepmag"] - df["M_REAL"] + 5) / 5)) * 3.26156



#coordinates
df["x"] = df["distance"] * np.cos(df["ra"]) * np.cos(df["dec"])
df["y"] = df["distance"] * np.sin(df["ra"]) * np.cos(df["dec"])
df["z"] = df["distance"] * np.sin(df["dec"])



df["map"] = df["koi_disposition"].copy()
df.loc[habitableOnesV.index, "map"] = "Habitable Candidate Planet"
df.loc[habitableOnes.index, "map"] = "Habitable Confirmed Planet"
df["displayname"] = df["kepler_name"].fillna(df["kepoi_name"])

df.loc["earth", "x"] = 0
df.loc["earth", "y"] = 0
df.loc["earth", "z"] = 0
df.loc["earth", "koi_period"] = 365.25
df.loc["earth", "koi_prad"] = 1
df.loc["earth", "koi_insol"] = 1
df.loc["earth", "koi_steff"] = 5778
df.loc["earth", "distance"] = 0
df.loc["earth", "displayname"] = "Earth"
df.loc["earth", "map"] = "Earth"


import plotly.express as pl

fig = pl.scatter_3d(
    df,
    x = "x",
    y = "y",
    z = "z",
    color = "map",
    color_discrete_map = {
        "CONFIRMED": "#5558ed",
        "FALSE POSITIVE": "#3e3535",
        "CANDIDATE": "#eda655",
        "Habitable Confirmed Planet": "#00ff44",
        "Habitable Candidate Planet": "#b3ff00",
        "Earth": "#ffffff",
    },
    hover_name = "displayname",
    hover_data = {
        "x" : False,
        "y" : False,
        "z" : False,
        "koi_period": ":.2f",
        "koi_prad": ":.2f",
        "koi_insol": ":.2f",
        "koi_steff": ":.0f",
        "distance": ":.0f"
    },
    labels = {
        "koi_period": "Orbital Period (Earth Days)",
        "koi_prad": "Planet Radius (Radius of Earth)",
        "koi_insol": "Stellar Flux (Flux of Earth)",
        "koi_steff": "Stellar Effective Temperature (Kelvin)",
        "distance": "Distance (Light Year)"
    },
    title = "Kepler Objects of Interest Visualization - 3D Map for Habitable and Possible Planets",
    opacity = 0.8
)

fig.update_traces(marker=dict(size=2.3))
fig.update_layout(
    template = "plotly_dark",
    scene = dict(xaxis_title="X (Light Year)",
        yaxis_title="Y (Light Year)",
        zaxis_title="Z (Light Year)",
        aspectmode="data"
    ),
    legend_title_text = "Target Class"
)

fig.show()