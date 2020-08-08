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

#external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
#external_stylesheets=external_stylesheets
app = dash.Dash(__name__)

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
                                        dcc.Dropdown(id='year',value=2020,clearable=False,searchable=False),
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
                                        dcc.Dropdown(id='driver1',value='hamilton',clearable=False,searchable=False),
                                    ],
                                ),
                                html.Div(
                                    className="div-for-dropdown",
                                    children=[
                                        dcc.Dropdown(id='driver2',value='bottas',clearable=False,searchable=False),
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
                    ],
                ),
            ],
        )
    ]
)


@app.callback(
    Output(component_id='my-output2', component_property='figure'),
    [Input(component_id='year', component_property='value'),Input(component_id='race_name', component_property='value'),Input(component_id='driver1', component_property='value'),Input(component_id='driver2', component_property='value')]
)
def update_delta_graph(year,race_name,driver1,driver2):
    return plotRaceComparisonGraph(year,race_name,driver1,driver2)

@app.callback(
    Output(component_id='my-output', component_property='figure'),
    [Input(component_id='year', component_property='value'), Input(component_id='race_name', component_property='value')]
)
def update_output_div(year, race_name):
    return plotRaceGraph(year, race_name)

@app.callback(
    [Output(component_id='driver1', component_property='options'),Output(component_id='driver2', component_property='options')],
    [Input(component_id='year', component_property='value'), Input(component_id='race_name', component_property='value')]
)
def update_output_div(year, race_name):
    drivers = getDriverNames(year, race_name)
    print(drivers)
    return [{'label': i, 'value': i} for i in drivers],[{'label': i, 'value': i} for i in drivers]

# Import all the data
drivers_df = pd.read_csv("./f1db_csv/drivers.csv").drop(columns = "url")
lap_times_df = pd.read_csv("./f1db_csv/lap_times.csv")
results_df = pd.read_csv("./f1db_csv/results.csv")
constructors_df = pd.read_csv("./f1db_csv/constructors.csv")
races_df = pd.read_csv("./f1db_csv/races.csv")

# Clean some names and create new variables
# drivers_df
drivers_df["number"] = drivers_df["number"].replace({r"\N": None})
drivers_df["driverName"] = drivers_df["forename"].str.cat(drivers_df["surname"],sep = " ")
drivers_df = drivers_df.drop(columns = ["forename", "surname"])

# lap_times_df
clean_lt_df = lap_times_df[["raceId", "driverId", "lap", "milliseconds"]]
clean_lt_df["seconds"] = clean_lt_df.milliseconds / 1000
clean_lt_df = clean_lt_df.drop(columns = "milliseconds")


def plotRaceGraph(year, race_name):
    races_temp = races_df[races_df.year == year]
    race_id = int(races_temp.raceId[races_temp.name == race_name])
    lap_times_1 = clean_lt_df[clean_lt_df.raceId == race_id]
    results_1 = results_df[results_df.raceId == race_id]
    df_1 = pd.merge(drivers_df[["driverId", "driverName", "driverRef", "number"]], lap_times_1, on = "driverId")
    df_2 = pd.merge(df_1, results_1[["resultId", "driverId", "constructorId", "position"]], on = "driverId")
    df_3 = pd.merge(df_2, constructors_df[["constructorId", "constructorRef"]], on = "constructorId")
    df_3["constructorRef"] = df_3["constructorRef"].str.title()
    df_4 = pd.merge(df_3, races_df[["raceId", "year", "name"]], on = "raceId")

    fig = px.line(df_4, x = "lap", y = "seconds", color = "driverName", hover_name = "driverName", hover_data = {"driverName" : False, "constructorRef" : True})

    fig.update_layout(plot_bgcolor="#323130",
        paper_bgcolor="#323130",font=dict(color="white"))

    return fig

def plotRaceComparisonGraph(year, race_name, driver1, driver2):
    races_temp = races_df[races_df.year == year]
    race_id = int(races_temp.raceId[races_temp.name == race_name])
    lap_times_1 = clean_lt_df[clean_lt_df.raceId == race_id]
    results_1 = results_df[results_df.raceId == race_id]
    df_1 = pd.merge(drivers_df[["driverId", "driverName", "driverRef", "number"]], lap_times_1, on = "driverId")
    df_2 = pd.merge(df_1, results_1[["resultId", "driverId", "constructorId", "position"]], on = "driverId")
    df_3 = pd.merge(df_2, constructors_df[["constructorId", "constructorRef"]], on = "constructorId")
    df_3["constructorRef"] = df_3["constructorRef"].str.title()
    df_4 = pd.merge(df_3, races_df[["raceId", "year", "name"]], on = "raceId")

    df5 = df_4[df_4.driverRef==driver1]["seconds"].rename('driver1')
    df6 = df_4[df_4.driverRef==driver2]["seconds"].rename('driver2')
    df5 = df5.reset_index(drop=True)
    df6 = df6.reset_index(drop=True)

    df6 = pd.concat([df5,df6],axis=1)
	# df4['driver2'] = df_3[df_3.driverRef==driver2].seconds
    df6['delta'] = df6.driver1-df6.driver2
    df6['lap'] = df6.index + 1

    fig = tls.make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Scatter(x = df6['lap'], y = df6['delta'],mode='lines',name='Delta'),secondary_y=False)
    fig.add_trace(go.Scatter(x = df6['lap'], y = df6['driver1'],mode='lines',name=driver1),secondary_y=True)
    fig.add_trace(go.Scatter(x = df6['lap'], y = df6['driver2'],mode='lines',name=driver2),secondary_y=True)

	#set line colors
	#fig['data'][0]['line']['color']="#00ff00"
    fig['layout']['yaxis2']['showgrid'] = False

    fig.update_layout(plot_bgcolor="#323130",
        paper_bgcolor="#323130",font=dict(color="white"))

    return fig

def getDriverNames(year, race_name):
    races_temp = races_df[races_df.year == year]
    race_id = int(races_temp.raceId[races_temp.name == race_name])
    lap_times_1 = clean_lt_df[clean_lt_df.raceId == race_id]
    results_1 = results_df[results_df.raceId == race_id]
    df_1 = pd.merge(drivers_df[["driverId", "driverName", "number"]], lap_times_1, on = "driverId")
    df_2 = pd.merge(df_1, results_1[["resultId", "driverId", "constructorId", "position"]], on = "driverId")
    df_3 = pd.merge(df_2, constructors_df[["constructorId", "constructorRef"]], on = "constructorId")
    df_3["constructorRef"] = df_3["constructorRef"].str.title()

    return df_3.driverName.unique()

if __name__ == '__main__':
    app.run_server(debug=True)