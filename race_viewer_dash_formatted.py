#race_viewer_dash
import numpy as np
import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly import tools as tls
from dash.dependencies import Input, Output
import datetime as dt

app = dash.Dash(__name__)

# Import all the data
drivers_df = pd.read_csv("./f1db_csv/drivers.csv").drop(columns = "url")
lap_times_df = pd.read_csv("./f1db_csv/lap_times.csv")
results_df = pd.read_csv("./f1db_csv/results.csv")
constructors_df = pd.read_csv("./f1db_csv/constructors.csv")
races_df = pd.read_csv("./f1db_csv/races.csv")
colors_df = pd.read_csv("./colors.csv")

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

class dataContainer:
    def __init__(self):
        self.race_table,self.year_races,self.color_palette = create_race_table(2020, "British Grand Prix")

    def plotRaceGraph(self):
        fig = px.line(self.race_table, x = "lap", y = "seconds", color = "driverName", hover_name = "driverName", hover_data = {"driverName" : False, "constructorRef" : True}, 
            color_discrete_map = self.color_palette)
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
        df_3['delta'] = df_3.driver1-df_3.driver2
        df_3['lap'] = df_3.index + 1

        fig = tls.make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(go.Scatter(x = df_3['lap'], y = df_3['delta'],mode='lines',name='Delta'),secondary_y=False)
        fig.add_trace(go.Scatter(x = df_3['lap'], y = df_3['driver1'],mode='lines',name=driver1),secondary_y=True)
        fig.add_trace(go.Scatter(x = df_3['lap'], y = df_3['driver2'],mode='lines',name=driver2),secondary_y=True)

        fig['layout']['yaxis2']['showgrid'] = False

        fig.update_layout(plot_bgcolor="#323130",
            paper_bgcolor="#323130",font=dict(color="white"))

        return fig

    def plotDeltaGraph(self):
        df_grouped = [y for x,y in self.race_table.groupby('driverName',as_index=False)]
        for i in range(len(df_grouped)):
            df_driver = df_grouped[i]
            df_grouped[i] = pd.concat([df_driver[["lap","driverName"]],df_driver["seconds"].cumsum()],axis=1)
            df_grouped[i] = df_grouped[i].reset_index(drop=True)

        #get minimum cumulative time for each lap

        df_merged = pd.concat(df_grouped)
        print(df_merged)

        df_minimum = [y for x,y in df_merged.groupby('lap',as_index=False)]

        for i in range(len(df_minimum)):
            df_minimum[i]['min'] = df_minimum[i]['seconds'].min()
            #df_minimum[i]['min'] = df_minimum[i]['milliseconds'][df_minimum[i]['driverRef']=='bottas']
            df_minimum[i]['delta'] = df_minimum[i]['seconds']-df_minimum[i]['min']
            df_minimum[i] = df_minimum[i][['lap','driverName','delta']]

        df_deltas = pd.concat(df_minimum,ignore_index=True)

        fig = px.line(df_deltas,x="lap",y="delta",color='driverName')
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

#dataContainer object creation
dataContainer = dataContainer()

app.layout = html.Div(
    children=[
        html.Div(
            className="row",
            children=[
                # Column for user controls
                html.Div(
                    className="four columns div-user-controls",
                    children=[
                        html.H2("F1Data.io"),
                        html.P(
                            """Select different races and compare drivers."""
                        ),
                        html.Div(
                                    className="div-for-dropdown",
                                    children=[
                                        dcc.Dropdown(id='year',value=2020,clearable=False,searchable=False,options=[{'label': i, 'value': i} for i in races_df['year'].unique()]),
                                    ],
                                ),
                        html.Div(
                                    className="div-for-dropdown",
                                    children=[
                                        dcc.Dropdown(id='race_name',value='British Grand Prix',clearable=False,searchable=False),
                                    ],
                                ),
                        # Change to side-by-side for mobile layout
                        html.Div(
                            className="row",
                            children=[
                                html.Div(
                                    className="div-for-dropdown",
                                    children=[
                                        dcc.Dropdown(id='driver1',value='Lewis Hamilton',clearable=False,searchable=False),
                                    ],
                                ),
                                html.Div(
                                    className="div-for-dropdown",
                                    children=[
                                        dcc.Dropdown(id='driver2',value='Valtteri Bottas',clearable=False,searchable=False),
                                    ],
                                ),
                            ],
                        ),
                        
                    ],
                ),
                # Column for app graphs and plots
                html.Div(
                    className="eight columns div-for-charts bg-grey",
                    children=[
                        dcc.Graph(id='my-output'),
                        html.Div(
                            className="text-padding",
                            children=[
                                "Select two drivers to view their relative lap times"
                            ],
                        ),
                        dcc.Graph(id='my-output2'),
                        dcc.Graph(id='deltaGraph',figure = dataContainer.plotDeltaGraph())
                    ],
                ),
            ],
        )
    ]
)


@app.callback(
    Output(component_id='my-output2', component_property='figure'),
    [Input(component_id='driver1', component_property='value'),Input(component_id='driver2', component_property='value')]
)
def update_delta_graph(driver1,driver2):
    return dataContainer.plotRaceComparisonGraph(driver1,driver2)

@app.callback(
    [Output(component_id='my-output', component_property='figure'),Output(component_id='driver1', component_property='options'),Output(component_id='driver2', component_property='options'),Output(component_id='race_name', component_property='options')],
    [Input(component_id='year', component_property='value'), Input(component_id='race_name', component_property='value')]
)
def update_race_graph(year, race_name):
    dataContainer.race_table,dataContainer.year_races,dataContainer.color_palette = create_race_table(year,race_name)
    drivers = dataContainer.getDriverNames()
    race_names = dataContainer.getRaceNames()
    print(race_names)
    return dataContainer.plotRaceGraph(),[{'label': i, 'value': i} for i in drivers],[{'label': i, 'value': i} for i in drivers],[{'label': i, 'value': i} for i in race_names]

if __name__ == '__main__':
    app.run_server(debug=True)