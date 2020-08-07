#!/usr/bin/env python
# coding: utf-8

# # Dashboard for F1 Data Insights

# ### Import Packages

# In[19]:


import numpy as np
import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import plotly.express as px

# View all rows and columns of a dataframe
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


# ## Loading in Data

# In[3]:


# Load Drivers data
drivers_df = pd.read_csv("./f1db_csv/drivers.csv")
drivers_df


# In[4]:


# Load lap times data
lap_times_df = pd.read_csv("./f1db_csv/lap_times.csv")


# In[14]:


# Load results data
results_df = pd.read_csv("./f1db_csv/results.csv")

# Filter to only race 1034
results_1034_df = results_df[results_df.raceId == 1034]


# In[16]:


# Load constructors names
constructors_df = pd.read_csv("./f1db_csv/constructors.csv")


# In[54]:


clean_lt_df = lap_times_df[["raceId", "driverId", "lap", "milliseconds"]]
race_1034_df = clean_lt_df[clean_lt_df.raceId == 1034]
race_1034_df["seconds"] = race_1034_df.milliseconds / 1000
race_1034_df = race_1034_df.drop(columns = "milliseconds")
race_1034_df


# In[55]:


df_1 = pd.merge(race_1034_df, drivers_df[["driverId", "driverRef", "number"]], on = "driverId")
df_2 = pd.merge(df_1, results_1034_df[["resultId", "driverId", "constructorId"]], on = "driverId")
df_3 = pd.merge(df_2, constructors_df[["constructorId", "constructorRef"]], on = "constructorId")
df_3


# In[68]:


# Create table with driver, team, and team color
driver_ref_table = df_3[["driverRef", "constructorRef"]].drop_duplicates()
driver_ref_table.loc[10, :] = ["perez", "racing_point"]
driver_ref_table = driver_ref_table.sort_values(by = "constructorRef")
driver_ref_table = driver_ref_table.reset_index(drop = True)
driver_ref_table


# In[70]:


px.line(df_3, x = "lap", y = "seconds", color = "driverRef", color_discrete_sequence = color_pal, 
        hover_name = "driverRef", hover_data = {"driverRef" : False, "constructorRef" : True},
        color_discrete_map = {"hamilton" : "#00D2BE", "bottas" : "#00D2BE", 
                             "max_verstappen" : "#1E41FF", "albon" : "#1E41FF",
                             "leclerc" : "#DC0000", "vettel" : "#DC0000",
                             "sainz" : "#FF8700", "norris" : "#FF8700",
                             "ricciardo" : "#FFF500", "ocon" : "#FFF500",
                             "stroll" : "#F596C8", 
                             "gasly": "#469BFF", "kvyat" : "#469BFF",
                             "raikkonen" : "#9B0000", "giovinazzi" : "#9B0000",
                             "grosjean" : "#F0D787", "kevin_magnussen" : "#F0D787",
                             "latifi" : "white", "russell" : "white"})


# In[ ]:




