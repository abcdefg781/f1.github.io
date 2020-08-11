import pandas as pd
import copy
import warnings
import datetime as dt
import plotly.express as px

history_df = pd.read_csv("./f1db_csv/driver_history.csv")

driver_min_df = history_df[["driverName", "year", "name", "date", "constructorName", "minQualifyingTime", "fastestLapTime"]]
min_qual_times = driver_min_df.groupby("date")["minQualifyingTime"].min()
min_race_times = driver_min_df.groupby("date")["fastestLapTime"].min()
min_times_df = pd.merge(min_qual_times, min_race_times, on = "date")
print(min_times_df)