import dash
import dash_html_components as html
import plotly.graph_objects as go
import dash_core_components as dcc
import plotly.express as px
from dash.dependencies import Input, Output
import pandas as pd

app = dash.Dash(__name__)




df = pd.read_csv('data/covid_data_train.csv')


data=go.Scattermapbox(
    lon=df['lng'],
    lat=df['lat'],
    text=df['name'],
    mode='markers'
)
mapbox_access_token = 'your-free-token'

layout = dict(margin=dict(l=0, t=0, r=0, b=0, pad=0),
              mapbox_style="open-street-map")

fig = go.Figure(data=data, layout=layout)


app.layout = html.Div(children=[
    html.H1(children='Identified Geothermal Systems of the Western USA'),
    html.Div(children='''
        This data was provided by the USGS.
    '''),

    dcc.Graph(
        id='example-map',
        figure=fig
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)

