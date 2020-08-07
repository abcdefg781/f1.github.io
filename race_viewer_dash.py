#race_viewer_dash

import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly_express as px

df = pd.read_csv('./f1db_csv/lap_times.csv')
df_drivers = pd.read_csv('./f1db_csv/drivers.csv')
df2 = df[["raceId","driverId","lap","milliseconds"]]
df3 = df2[(df2["raceId"]==1034)]
df_merged = df3.merge(df_drivers[["driverId","driverRef"]],on='driverId')


def generate_table(dataframe, max_rows=10):
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), max_rows))
        ])
    ])


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

fig = px.line(df_merged,x='lap',y='milliseconds',color='driverRef')

app.layout = html.Div(children=[
    #html.H4(children='Race Table'),
    #generate_table(df_merged),
    dcc.Graph(id='racegraph',figure=fig)
])

if __name__ == '__main__':
    app.run_server(debug=True)