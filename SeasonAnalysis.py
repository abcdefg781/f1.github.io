# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import copy
import warnings
import datetime as dt
import numpy as np
import plotly
import plotly.express as px

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)


# %%
drivers_df = pd.read_csv("./f1db_csv/drivers.csv").drop(columns = "url")
races_df = pd.read_csv("./f1db_csv/races.csv")
constructors_df = pd.read_csv("./f1db_csv/constructors.csv")
races_df = pd.read_csv("./f1db_csv/races.csv")
laptimes_df = pd.read_csv("./f1db_csv/lap_times.csv")
results_df = pd.read_csv("./f1db_csv/results.csv")

drivers_df["number"] = drivers_df["number"].replace({r"\N": None})
drivers_df["driverName"] = drivers_df["forename"].str.cat(drivers_df["surname"],sep = " ")
drivers_df = drivers_df.drop(columns = ["forename", "surname"])

# results_df
results_df["position"] = results_df["position"].replace({r"\N": None})

standings_df = pd.read_csv("./f1db_csv/driver_standings.csv")
driver_history_df = pd.read_csv("./f1db_csv/driver_history.csv")
colors_df = pd.read_csv("./f1db_csv/constructors_colors.csv")
colors_df.rename(columns={"name":"constructorName"},inplace=True)



# %%
selectedSeason = 2021

def plotYearViolin(year):
    races_season = races_df[races_df["year"]==year]
    raceid_list = races_season["raceId"]

    #filter out lap times of the selected races
    laptimes_df = laptimes_df[laptimes_df['raceId'].isin(raceid_list)]
    driver_history_race = driver_history_df[driver_history_df['raceId'].isin(raceid_list)]
    driver_history_race = pd.merge(driver_history_race,colors_df,on="constructorName")
    
    max_raceid = laptimes_df["raceId"].unique().max()
    race_standings = standings_df[standings_df['raceId']==max_raceid]
    race_standings.drop(columns=["driverStandingsId","raceId","points","wins","positionText"],inplace=True)
    

    drivers = pd.DataFrame(laptimes_df["driverId"].unique(),columns=["driverId"])
    drivers = pd.merge(drivers,race_standings,on="driverId")
    drivers.sort_values('position',inplace=True)

    fig = go.Figure()
    for driver in drivers["driverId"]:
        driver_entry = drivers_df[drivers_df["driverId"]==driver]
        driver_name = driver_entry['driverName']
        driver_name = driver_name.tolist()[0]
        driver_laptimes_season = laptimes_df[laptimes_df["driverId"]==driver]
        drivers_races_season = driver_history_race[driver_history_race["driverName"]==driver_name]
        max_raceid = drivers_races_season["raceId"].max()
        color = drivers_races_season[drivers_races_season["raceId"]==max_raceid]["color"]
        color = color.tolist()[0]
        driver_laptimes_season.sort_values('position',inplace=True)
        violin = go.Violin(y=driver_laptimes_season["position"],bandwidth=0.5,points=False,scalemode="width",scalegroup="group",meanline_visible=True,opacity=0.8,hoveron="violins",x0=driver_name,name=driver_name,line_color='black',fillcolor=color)
        fig.add_trace(violin)
    fig.update_yaxes(type='category')
    fig.update_yaxes(categoryorder='array', categoryarray= list(range(1,laptimes_df['position'].max(),1)))
    fig.update_traces(spanmode="hard",line_width=1)
    fig.update_layout(
        yaxis= dict(tickmode='linear',tick0=1,dtick=1)
    )
    fig.update_layout(plot_bgcolor="#323130",
                paper_bgcolor="#323130",font=dict(color="white"),
                yaxis_title="Position",
                title='Driver position for each lap',
                margin = dict(l=20,r=20,t=20,b=20)
                )
    return fig


