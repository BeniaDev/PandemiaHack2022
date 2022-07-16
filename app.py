import logging

import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import eli5
import pandas as pd
import plotly.graph_objects as go
from dash.dependencies import Input, Output

from src.model import load_models, predict

tree_model, knn_model = load_models()

def load_test_data(path):
    test_df = pd.read_csv(path)
    test_df['inf_rate'], X_test_data = predict(path)
    test_df['text'] = test_df.apply(lambda x: x['name']+ '  Уровень заражения: '+str(round(x['inf_rate'], 1)), axis= 1)
    return test_df, X_test_data

def get_explain_data(town_name, df, X):
    data = X[df['name']==town_name].iloc[0]
    explanation_pred = eli5.explain_prediction_df(estimator=tree_model, doc=data)
    return explanation_pred[['feature','weight']]

df, X_test_data = load_test_data('data/covid_data_test.csv')


logging.basicConfig(level=logging.INFO,
                    handlers=[
                        logging.FileHandler("./logs/app.log", "a"),
                        logging.StreamHandler()
                    ],
                    format='[%(asctime)s | %(levelname)s]: %(message)s',
                    datefmt='%m.%d.%Y %H:%M:%S')


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])


data=go.Scattermapbox(
    lon=df['lng'],
    lat=df['lat'],
    text=df['text'],
    mode='markers',
    marker={"size": 3*(df['inf_rate']+1),'color':'#ff3030'},
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
fig.update_layout(title_text='Карта предсказаний уровня заражения для городов России')

childrens=[
    html.H1(children='Уровень заражения Covid-19 на март 2020 года'),
    html.Br(),
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
    bar_df = get_explain_data(town_name, df, X_test_data)

    colors = ['lightslategray', ] * 5
    colors[1] = 'crimson'

    bar_df.loc[0,'feature'] = 'Базовое смещение'
    fig = go.Figure(data=[go.Bar(
        x=bar_df['feature'],
        y=bar_df['weight'],
        marker_color=['#ff3030' if w>0 else '#85ff30' for w in bar_df['weight']]  # marker color can be a single color value or an iterable
    )])
    fig.update_layout(title_text='Вклад признаков в предсказание моделью')
    return fig

if __name__ == '__main__':
    app.run_server(debug=False)

