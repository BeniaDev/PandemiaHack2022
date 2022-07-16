import dash
import dash_html_components as html
import plotly.graph_objects as go
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import plotly.express as px
from dash.dependencies import Input, Output
import dash_core_components as dcc
import pandas as pd


import logging
from src.model import load_models, predict

predict('./data/covid_data_test.csv')

logging.basicConfig(level=logging.INFO,
                    handlers=[
                        logging.FileHandler("./logs/app.log", "a"),
                        logging.StreamHandler()
                    ],
                    format='[%(asctime)s | %(levelname)s]: %(message)s',
                    datefmt='%m.%d.%Y %H:%M:%S')


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
df = pd.read_csv('data/covid_data_train.csv')
df = df[df['inf_rate'].notnull()]
df['text'] = df.apply(lambda x: x['name']+ '  Уровень заражения: '+str(round(x['inf_rate'], 1)), axis= 1)

data=go.Scattermapbox(
    lon=df['lng'],
    lat=df['lat'],
    text=df['text'],
    mode='markers',
    marker={"size": 3*(df['inf_rate']+1)},
)

layout = dict(margin=dict(
                        l=40,
                        r=40,
                        b=40,
                        t=40,
                        pad=0
                            ),
              autosize=False,
              mapbox_style="open-street-map",
              mapbox=dict(
                    center=dict(
                        lat=60,
                        lon=92
                    ),
                    zoom=1.7
                ))

fig = go.Figure(data=data, layout=layout)


childrens=[
    html.H1(children='Уровень заражения Covid-19'),
    html.Div(children='''
        This data was provided by the USGS.
    '''),

    html.Div([
        dcc.Graph(
            id='covid-map',
            figure=fig
        )
    ], style={'width': '59%', 'display': 'inline-block', 'padding': '0 0'}),
    html.Div([
        dcc.Graph(id='town-graph')
    ], style={'width': '39%', 'display': 'inline-block', 'padding': '0 0'})
]

body = dbc.Container([
dbc.Row(
            childrens, justify="center", align="center", className="h-50"
            )
],style={"height": "100vh"}

)
app.layout = html.Div([body])

@app.callback(
    Output('town-graph', 'figure'),
    Input('covid-map', 'clickData'))
def update_y_timeseries(hoverData):
    town_name = hoverData['points'][0]['text'].split('  Уровень заражения')[0]

    colors = ['lightslategray', ] * 5
    colors[1] = 'crimson'
    fig = go.Figure(data=[go.Bar(
        x=['Feature A', 'Feature B', 'Feature C',
           'Feature D', 'Feature E'],
        y=[20, 14, 23, 25, 22],
        marker_color=colors  # marker color can be a single color value or an iterable
    )])
    fig.update_layout(title_text='Least Used Feature')
    return fig


if __name__ == '__main__':
    app.run_server(debug=True)

    #print(predict("./data/Тестовый датасет/covid_data_test.csv"))