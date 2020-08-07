#race_viewer_dash

import numpy as np
import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
from dash.dependencies import Input, Output

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
# dcc.Graph(id='racegraph',figure=fig)

app.layout = html.Div([
    html.H6("Change the value in the text box to see callbacks in action!"),
    html.Div(["Input: ",
              dcc.Input(id='my-input', value=1034, type='number')]),
    html.Br(),
    #html.Div(id='my-output'),
    dcc.Graph(id='my-output')
])


@app.callback(
    Output(component_id='my-output', component_property='figure'),
    [Input(component_id='my-input', component_property='value')]
)
def update_output_div(input_value):
    return plotRaceGraph(input_value)

def plotRaceGraph(raceNum):
	# Load Drivers data
	drivers_df = pd.read_csv("./f1db_csv/drivers.csv")

	# Load lap times data
	lap_times_df = pd.read_csv("./f1db_csv/lap_times.csv")

	# Load results data
	results_df = pd.read_csv("./f1db_csv/results.csv")

	# Filter to only race 1034
	results_1034_df = results_df[results_df.raceId == raceNum]

	# Load constructors names
	constructors_df = pd.read_csv("./f1db_csv/constructors.csv")

	clean_lt_df = lap_times_df[["raceId", "driverId", "lap", "milliseconds"]]
	race_1034_df = clean_lt_df[clean_lt_df.raceId == raceNum]
	race_1034_df["seconds"] = race_1034_df.milliseconds / 1000
	race_1034_df = race_1034_df.drop(columns = "milliseconds")

	df_1 = pd.merge(race_1034_df, drivers_df[["driverId", "driverRef", "number"]], on = "driverId")
	df_2 = pd.merge(df_1, results_1034_df[["resultId", "driverId", "constructorId"]], on = "driverId")
	df_3 = pd.merge(df_2, constructors_df[["constructorId", "constructorRef"]], on = "constructorId")

	# Create table with driver, team, and team color
	driver_ref_table = df_3[["driverRef", "constructorRef"]].drop_duplicates()
	driver_ref_table.loc[10, :] = ["perez", "racing_point"]
	driver_ref_table = driver_ref_table.sort_values(by = "constructorRef")
	driver_ref_table = driver_ref_table.reset_index(drop = True)

	fig = px.line(df_3, x = "lap", y = "seconds", color = "driverRef", 
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
	                             "latifi" : "white", "russell" : "white"},width=1000,height=600)
	return fig

if __name__ == '__main__':
    app.run_server(debug=True)