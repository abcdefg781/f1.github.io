import pandas as pd
import numpy as np
import plotly.express as px
import plotly
import plotly.graph_objects as go

# Import data and Functions
# Import all the data
drivers_df = pd.read_csv("./f1db_csv/drivers.csv").drop(columns = "url")
lap_times_df = pd.read_csv("./f1db_csv/lap_times.csv")
results_df = pd.read_csv("./f1db_csv/results.csv")
constructors_df = pd.read_csv("./f1db_csv/constructors.csv")
races_df = pd.read_csv("./f1db_csv/races.csv")
driver_history_df = pd.read_csv("./f1db_csv/driver_history.csv")
constructor_colors_df = pd.read_csv("./f1db_csv/constructors_colors.csv")

# Clean some names and create new variables
# drivers_df
drivers_df["number"] = drivers_df["number"].replace({r"\N": None})
drivers_df["driverName"] = drivers_df["forename"].str.cat(drivers_df["surname"],sep = " ")
drivers_df = drivers_df.drop(columns = ["forename", "surname"])

# lap_times_df
clean_lt_df = lap_times_df[["raceId", "driverId", "lap", "milliseconds"]]
clean_lt_df["seconds"] = clean_lt_df.milliseconds / 1000
clean_lt_df = clean_lt_df.drop(columns = "milliseconds")

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
    colored_df = pd.merge(df_4, constructor_colors_df[["constructorId", "color"]], on = "constructorId")
    color_palette = pd.Series(colored_df.color.values, index = colored_df.driverName).to_dict()
    return df_4, races_temp, color_palette,colored_df

race_table,races_temp,color_palette,color_df = create_race_table(2021,'Hungarian Grand Prix')
print(color_df)
def plotDeltaGraph(deltaType):
        df_grouped = [y for x,y in race_table.groupby('driverName',as_index=False)]
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
        df_deltas = df_deltas.merge(race_table[['driverName','constructorRef']],on='driverName')
        df_deltas = df_deltas.sort_values(by=["constructorRef","driverName"])

        df_team_grouped = [y for x,y in df_deltas.groupby('constructorRef',as_index=False)]

        fig = go.Figure()
        for i in range(len(df_team_grouped)):
            df_team_grouped[i].loc[df_team_grouped[i]['delta'].abs()>500] = np.nan
            df_grouped = [y for x,y in df_team_grouped[i].groupby('driverName',as_index=False)]
            for j in range(len(df_grouped)):
                df_driver = df_grouped[j]
                name = df_driver['driverName'].iloc[0]
                print(name)
                color = color_df[color_df.driverName==name]['color'].iloc[0]
                if j == 0:
                    line = go.Scatter(x=df_driver['lap'],y=df_driver['delta'],name=name,mode='lines',line=go.scatter.Line(color=color))
                else:
                    line = go.Scatter(x=df_driver['lap'],y=df_driver['delta'],name=name,mode='lines',line=go.scatter.Line(color=color,dash='dot'))
                fig.add_trace(line)
        fig.update_layout(plot_bgcolor="#323130",
            paper_bgcolor="#323130",font=dict(color="white"),
            xaxis_title="Lap",
            yaxis_title="Delta (s)"
            )
        #fig.update_layout(xaxis=dict(range=[1,64.9]))
        fig.update_layout(
            title={
                'text': "Relative to first place",
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'})
        fig.update_yaxes(automargin=True,autorange="reversed")
        fig.write_image("fig3.jpg",width=1200,height=600,scale=5)
        fig.show()
        #plotly.offline.plot(fig, filename= "Deltaplot2")

plotDeltaGraph('min')

