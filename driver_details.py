import time

import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from dash import Dash
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly import tools as tls
from dash.dependencies import Input, Output, State
import datetime as dt
import copy
import warnings

BS = "https://stackpath.bootstrapcdn.com/bootswatch/4.5.0/lux/bootstrap.min.css"
app = dash.Dash(external_stylesheets = [dbc.themes.BOOTSTRAP])
app.config['suppress_callback_exceptions'] = True
warnings.filterwarnings("ignore")

# Import data and Functions
# Import all the data
drivers_df = pd.read_csv("./f1db_csv/drivers.csv").drop(columns = "url")
lap_times_df = pd.read_csv("./f1db_csv/lap_times.csv")
results_df = pd.read_csv("./f1db_csv/results.csv")
constructors_df = pd.read_csv("./f1db_csv/constructors.csv")
races_df = pd.read_csv("./f1db_csv/races.csv")
colors_df = pd.read_csv("./colors.csv")
driver_history_df = pd.read_csv("./f1db_csv/driver_history.csv")

# Clean some names and create new variables
# drivers_df
drivers_df["number"] = drivers_df["number"].replace({r"\N": None})
drivers_df["driverName"] = drivers_df["forename"].str.cat(drivers_df["surname"],sep = " ")
drivers_df = drivers_df.drop(columns = ["forename", "surname"])

# lap_times_df
clean_lt_df = lap_times_df[["raceId", "driverId", "lap", "milliseconds"]]
clean_lt_df["seconds"] = clean_lt_df.milliseconds / 1000
clean_lt_df = clean_lt_df.drop(columns = "milliseconds")

# results_df
results_df["position"] = results_df["position"].replace({r"\N": 99})
results_df["position"] = results_df["position"].astype(int)

# def create_driver_table(driver_name_1):
#     driver_1_table = driver_history_df[driver_history_df.driverName == driver_name_1].sort_values("date", ascending=False)
#     driver_1_table = driver_1_table.drop(driver_1_table.columns[0:4], axis=1)
#     driver_1_table.columns = ["Year", "Race Name", "Date", "Constructor", "Qualifying Position", "Fastest Qualifying Time", "Final Race Positiion", 
#         "Fastest Lap Time (Race)", "Cumulative Season Wins", "Cumulative Season Points", "Driver Standing After Race"]
#     return driver_1_table

def create_race_table(year, race_name):
    races_temp = races_df[races_df.year == year]
    race_id = int(races_temp.raceId[races_temp.name == race_name])
    lap_times_1 = clean_lt_df[clean_lt_df.raceId == race_id]
    results_1 = results_df[results_df.raceId == race_id]
    df_1 = pd.merge(drivers_df[["driverId", "driverName", "number"]], lap_times_1, on = "driverId")
    df_2 = pd.merge(df_1, results_1[["resultId", "driverId", "constructorId", "position"]], on = "driverId")
    df_3 = pd.merge(df_2, constructors_df[["constructorId", "constructorRef"]], on = "constructorId")
    df_3["constructorRef"] = df_3["constructorRef"].str.title()
    df_4 = pd.merge(df_3, races_df[["raceId", "year", "name"]], on = "raceId")
    df_4 = df_4.sort_values(by = ["position", "lap"])
    filtered_ref_df = colors_df[colors_df.year == year]
    color_palette = pd.Series(filtered_ref_df.color.values, index = filtered_ref_df.driverName).to_dict()
    return df_4, races_temp, color_palette

def create_driver_table(driver_name_1):
    driver_1_table = driver_history_df[driver_history_df.driverName == driver_name_1].sort_values("date", ascending=False)
    driver_1_table = driver_1_table.drop(driver_1_table.columns[0:4], axis=1)
    # driver_1_table.columns = ["Year", "Race Name", "Date", "Constructor", "Qualifying Position", "Fastest Qualifying Time", "Final Race Positiion", 
    #     "Fastest Lap Time (Race)", "Cumulative Season Wins", "Cumulative Season Points", "Driver Standing After Race"]
    return driver_1_table

def create_driver_table_year(driver_name_1, year):
    driver_1_table = driver_history_df[driver_history_df.driverName == driver_name_1].sort_values("date", ascending=False)
    driver_1_table = driver_1_table.drop(driver_1_table.columns[0:4], axis=1)
    driver_1_table = driver_1_table[driver_1_table.year == year]
    return driver_1_table

class dataContainer:
    def __init__(self):
        self.race_table,self.year_races,self.color_palette = create_race_table(2020, "British Grand Prix")
        self.driver_history_table = create_driver_table("Lewis Hamilton")
        self.driver_yr_history_table = create_driver_table("Lewis Hamilton")
    def plotRaceGraph(self):
        fig = px.line(self.race_table, x = "lap", y = "seconds", color = "driverName", hover_name = "driverName", hover_data = {"driverName" : False, "constructorRef" : True}, 
            # color_discrete_map = self.color_palette
            )
        fig.update_layout(legend_title_text=None)

        fig.update_layout(plot_bgcolor="#323130",
            paper_bgcolor="#323130",font=dict(color="white"))

        return fig

    def plotRaceComparisonGraph(self, driver1, driver2):
        df_1 = self.race_table[self.race_table.driverName==driver1]["seconds"].rename('driver1')
        df_2 = self.race_table[self.race_table.driverName==driver2]["seconds"].rename('driver2')
        df_1 = df_1.reset_index(drop=True)
        df_2 = df_2.reset_index(drop=True)

        df_3 = pd.concat([df_1,df_2],axis=1)
        # df4['driver2'] = df_3[df_3.driverRef==driver2].seconds
        df_3['delta'] = df_3.driver2-df_3.driver1
        df_3['lap'] = df_3.index + 1

        df_1_summed = df_1.cumsum()
        df_2_summed = df_2.cumsum()
        df_3['cumdelta'] = df_2_summed-df_1_summed
        df_3['hovertext'] = [0]*len(df_3['cumdelta'])
        df_3['hovertext2'] = [0]*len(df_3['delta'])

        for i in range(len(df_3['cumdelta'])):
            time = '%.3f' % np.abs(df_3['cumdelta'][i])
            if df_3['cumdelta'][i]>0:
                df_3['hovertext'][i] = driver1+' is ahead of '+driver2+' by '+ time +' seconds'
            else:
                df_3['hovertext'][i] = driver1+' is behind '+driver2+' by '+time+' seconds'

        for i in range(len(df_3['delta'])):
            time = '%.3f' % np.abs(df_3['delta'][i])
            if df_3['delta'][i]>0:
                df_3['hovertext2'][i] = driver1+' is faster than '+driver2+' by '+ time +' seconds this lap'
            else:
                df_3['hovertext2'][i] = driver1+' is slower than '+driver2+' by '+time+' seconds this lap'

        fig = go.Figure()
        fig.add_trace(go.Scatter(x = df_3['lap'], y = df_3['delta'],mode='lines',name='Lap Delta',customdata=df_3['hovertext2'],hovertemplate=
            '<br>%{customdata}'))
        fig.add_trace(go.Scatter(x = df_3['lap'], y = df_3['cumdelta'],mode='lines',name='Cumulative Delta',customdata=df_3['hovertext'],hovertemplate=
            '<br>%{customdata}'))


        fig.update_layout(
            title='Gap from '+driver1+' to '+driver2,
            xaxis_title="Lap",
            yaxis_title="Delta (s)",
            hovermode="x unified"
            )
        # fig.add_trace(go.Scatter(x = df_3['lap'], y = df_3['delta'],mode='lines',name='Lap Delta'),secondary_y=False)
        # fig.add_trace(go.Scatter(x = df_3['lap'], y = df_3['cumdelta'],mode='lines',name='Cumulative Delta'),secondary_y=True)
        # fig.add_trace(go.Scatter(x = df_3['lap'], y = df_3['driver1'],mode='lines',name=driver1),secondary_y=True)
        # fig.add_trace(go.Scatter(x = df_3['lap'], y = df_3['driver2'],mode='lines',name=driver2),secondary_y=True)


        fig.update_layout(plot_bgcolor="#323130",
            paper_bgcolor="#323130",font=dict(color="white"))

        return fig

    def plotDeltaGraph(self,deltaType):
        df_grouped = [y for x,y in self.race_table.groupby('driverName',as_index=False)]
        for i in range(len(df_grouped)):
            df_driver = df_grouped[i]
            df_grouped[i] = pd.concat([df_driver[["lap","driverName"]],df_driver["seconds"].cumsum()],axis=1)
            df_grouped[i] = df_grouped[i].reset_index(drop=True)

        #get minimum cumulative time for each lap

        df_merged = pd.concat(df_grouped)

        df_minimum = [y for x,y in df_merged.groupby('lap',as_index=False)]

        for i in range(len(df_minimum)):
            if deltaType == 'median':
                df_minimum[i]['min'] = df_minimum[i]['seconds'].median()
            elif deltaType == 'min':
                df_minimum[i]['min'] = df_minimum[i]['seconds'].min()
            elif deltaType == 'max':
                df_minimum[i]['min'] = df_minimum[i]['seconds'].max()
            else:
                df_minimum[i]['min'] = df_minimum[i]['seconds'][df_minimum[i]['driverName']==deltaType]

            df_minimum[i]['delta'] = df_minimum[i]['seconds']-df_minimum[i]['min']
            df_minimum[i] = df_minimum[i][['lap','driverName','delta']]

        df_deltas = pd.concat(df_minimum,ignore_index=True)

        fig = px.line(df_deltas,x="lap",y="delta",color='driverName')
        fig.update_layout(plot_bgcolor="#323130",
            paper_bgcolor="#323130",font=dict(color="white"),
            xaxis_title="Lap",
            yaxis_title="Delta (s)"
            )
        return fig

    def getDriverNames(self):
        return self.race_table.driverName.unique()

    def getRaceNames(self):
        today = pd.to_datetime('today')
        year_races = self.year_races[["raceId","year","name","date"]]
        year_races['datetimes'] = pd.to_datetime(year_races.date)
        year_races = year_races[year_races.datetimes<today]
        year_races = year_races.sort_values(by='datetimes',ascending=False)
        return year_races['name'].unique()

    def getAllDriverNames(self):
        return self.driver_history_table.driverName.unique()

#dataContainer object creation
dataContainer = dataContainer()

app.layout = dbc.Container(
    [
        dcc.Store(id="store"),
        html.Br(),
        html.Br(),
        html.H1("F1Data.io"),
        dbc.Tabs(
            [
                dbc.Tab(label = "Driver History", tab_id="collapses", children = [
                        html.Br(),
                        html.P("Select one driver to view their history."),
                        dcc.Dropdown(id='all_drivers', clearable=False,searchable=True,value='Lewis Hamilton'),
                        html.Br(),
                        html.Div(id='my-table')
                    ])
            ],
            id="tabs",
            active_tab="home",
        ),
        html.Div(id="tab-content", className="p-4"),
    ]
)


@app.callback(
    [Output(component_id='my-table', component_property='children'), Output(component_id='all_drivers', component_property='options')],
    [Input(component_id='all_drivers', component_property='value')]
)
def update_driver_history(driver_name_1):
    output = []
    dataContainer.driver_history_table = create_driver_table(driver_name_1)
    for i in range(len(dataContainer.driver_history_table.year.unique())): 
        year = dataContainer.driver_history_table.year.unique()[i]
        driver_table_year = dataContainer.driver_history_table[dataContainer.driver_history_table.year==year]
        print(year)
        output.append(
            html.Div(children = [
                html.Details([
                    html.Summary(year),
                    dash_table.DataTable(id='my-table'+str(i), columns=[{"name": j, "id": j} for j in driver_table_year.columns], 
                    data=driver_table_year.to_dict("records"),
                    style_as_list_view=True,
                    style_header={'backgroundColor': 'rgb(30, 30, 30)', 
                        'whiteSpace': 'normal',
                        'maxWidth': '120px',
                        'height': 'auto'},
                    style_cell={
                        'backgroundColor': 'rgb(50, 50, 50)',
                        'color': 'white',
                        'maxWidth': '120px'
                    },
                    style_table={'overflowX': 'auto'},
                    fixed_rows={'headers': True}
                    ),
                ]),

            ])
        )

    driver_names = driver_history_df.driverName.unique()
    all_drivers = [{'label': i, 'value': i} for i in driver_names]
    return output, all_drivers

# for j in range(1,3):
#     @app.callback(
#         [Output(component_id='my-table{}'.format(j+1), component_property='data'),Output(component_id='all_drivers{}'.format(j+1), component_property='options'),Output(component_id='year_list{}'.format(j+1), component_property='children')],
#         [Input(component_id='all_drivers', component_property='value'),Input(component_id='year', component_property='value')]
#         )
#     def update_driver_history(driver_name_1, year1):
#         driver_names = driver_history_df.driverName.unique()
#         all_drivers = [{'label': i, 'value': i} for i in driver_names]
#         driver_table = create_driver_table_year(driver_name_1)
#         year_list = driver_table.Year
#         for i in range(len(driver_table.Year)):
#             driver_table = driver_table[driver_table.Year == year1]
#             driver_table = driver_table.to_dict('records')
#         return driver_table, all_drivers, year_list
#     j += 1

# Load to Dash
if __name__ == "__main__":
    app.run_server(debug=True)
