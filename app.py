#Libraries
import pandas as pd

#Association Rule Mining
import lightgbm as lgb
from sklearn.metrics import classification_report, roc_curve, auc
import pickle

#Dash libraries
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
#import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import warnings
warnings.filterwarnings('ignore')

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

#read dataframe
df_news = pickle.load(open("test.pkl", "rb"))
train1 = pickle.load(open("train1.pkl", "rb"))

#read model
lgb_clf = pickle.load(open("lgb_model.pkl", "rb"))
xgb_model = pickle.load(open('xgboost2.pickle.dat', "rb"))

def generate_table(dataframe, max_rows=5):
    return html.Table(
        # Header
        [html.Tr([html.Th(col) for col in dataframe.columns])] +

        # Body
        [html.Tr([
            html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
        ]) for i in range(min(len(dataframe), max_rows))]
    )


app.layout = html.Div(children=[
    html.Div(style={'textAlign': "center"}, children=[
        html.H1("Viral News Prediction"),
        html.Div('''Which articles from Mashable.com will get Viral? What makes an article go Viral?''')]),
    html.Div(style={'textAlign': "center"},
             children=[html.H4(children='Data of News'),
             generate_table(df_news)]),
    #html.Div(style={'textAlign': "center"},
     #        children=[html.H4(children='Data of News'),
      #      generate_table(train1)]),
    html.Div(children=[
        html.H2("Model Evaluations"),
        html.H6('Please wait a moment after changing the values, and ignore the alerts until the table is updated'),
        html.Div(children = [
            html.Label('Select Model: '),
            dcc.Dropdown(
                id='model',
                options=[{'label':'Logistic Regression', 'value': 'logistic'},
                         {'label':'Random Forest Classifier', 'value': 'rf'},
                         {'label':'XGBoost Classifier', 'value': 'xgboost'},
                         {'label':'LightGBM Classifier', 'value': 'lightgbm'}],
                value='lightgbm'
            )
        ]),
        html.H3("LightGBM Parameters: "),
        html.Div(children = [
            html.Label('No. of Estimators: '),
            dcc.Slider(
                id='n_estimators',
                min=100,
                max=500,
                step=50,
                marks={
                        100: '100',
                        200: '200',
                        300: '300',
                        400: '400',
                        500: '500'
                    },
                value=415,
            ),

            html.Label('Learning Rate: '),
            dcc.Slider(
                id='learning_rate',
                min=0.01,
                max=0.5,
                step=0.01,
                marks={
                        0.01: '0.01',
                        0.02: '0.02',
                        0.05: '0.05',
                        0.1: '0.1',
                        0.3: '0.3',
                        0.5: '0.5',
                    },
                value=0.01,
            ),

            html.Label('Maximum Depth of Tree: '),
            dcc.Slider(
                id='max_depth',
                min=1,
                max=100,
                step=1,
                marks={
                    1: '1',
                    3: '3',
                    5: '5',
                    10: '10',
                    15: '15',
                    20: '20',
                    30: '30',
                    40: '40',
                    50: '50',
                    80: '80',
                    100: '100'
                },
                value=20,
            ),

            html.Label('Number of Leaves: '),
            dcc.Slider(
                id='num_leaves',
                min=5,
                max=50,
                step=1,
                marks={
                    5: '5',
                    15: '15',
                    20: '20',
                    30: '30',
                    50: '50'
                },
                value=15,
            ),

            html.Label('Subsample: '),
            dcc.Slider(
                id='subsample',
                min=0.1,
                max=1,
                step=0.1,
                marks={
                    100: '100',
                    200: '200',
                    300: '300',
                    400: '400',
                    500: '500'
                },
                value=0.4,
            ),

            html.Label('Regularisation Aplha: '),
            dcc.Slider(
                id='reg_alpha',
                min=0,
                max=1,
                step=0.1,
                value=0.2,
            ),

            html.Label('Regularisation Lambda: '),
            dcc.Slider(
                id='reg_lambda',
                min=0,
                max=1,
                step=0.1,
                value=0.1,
            ),

            html.Label('Column Sample by Tree: '),
            dcc.Slider(
                id='colsample_bytree',
                min=0.5,
                max=1,
                step=0.05,
                value=0.85,
            ),

            html.Label('Minimum Child Weight: '),
            dcc.Slider(
                id='min_child_weight',
                min=1,
                max=5,
                step=1,
                value=3,
            )
            ]),

        html.Div(id='report'),
        html.H4(children='ROC curve for Classification'),
        html.Div([dcc.Graph(id='indicator-graphic')])
        ])
])


@app.callback(
    Output('report','children'),
    [Input('n_estimators', 'value'), Input('learning_rate', 'value'),Input('max_depth','value'),Input('num_leaves','value'),
     Input('subsample','value'), Input('reg_alpha','value'), Input('reg_lambda','value'), Input('colsample_bytree','value'), Input('min_child_weight','value')]
    )
def evaluate(n_et, lr, md,nl,sb,al,lb,cs,cw):
    gbm = lgb.LGBMClassifier(boosting_type='gbdt', class_weight='balanced', n_estimators = n_et, learning_rate = lr,
                             max_depth=md, num_leaves=nl, subsample=sb, reg_alpha=al, reg_lambda=lb, colsample_bytree=cs, min_child_weight=cw)
    gbm.fit(train1.drop(['popular'], axis=1), train1['popular'])
    c_report = classification_report(df_news['popular'], gbm.predict(df_news.drop(['index','popular'], axis=1)), output_dict=True)
    return generate_table(pd.DataFrame(c_report).transpose(), max_rows=2)


@app.callback(
    Output('indicator-graphic','figure'),
    [Input('n_estimators', 'value'), Input('learning_rate', 'value'), Input('max_depth', 'value'),
     Input('num_leaves', 'value'),
     Input('subsample', 'value'), Input('reg_alpha', 'value'), Input('reg_lambda', 'value'),
     Input('colsample_bytree', 'value'), Input('min_child_weight', 'value')]
)
def update_graph(n_et, lr, md, nl, sb, al, lb, cs, cw):
    gbm = lgb.LGBMClassifier(boosting_type='gbdt', class_weight='balanced', n_estimators=n_et, learning_rate=lr,
                             max_depth=md, num_leaves=nl, subsample=sb, reg_alpha=al, reg_lambda=lb,
                             colsample_bytree=cs, min_child_weight=cw)
    gbm.fit(train1.drop(['popular'], axis=1), train1['popular'])
    proba = gbm.predict_proba(df_news.drop(['index','popular'], axis=1))
    fpr, tpr, _ = roc_curve(df_news['popular'], proba[:, 1])
    roc_auc = auc(fpr, tpr)
    lw = 2
    trace1 = go.Scatter(x=fpr, y=tpr,
                        mode='lines',
                        line=dict(color='darkorange', width=lw),
                        name='ROC curve (area = %0.2f)' % roc_auc
                        )

    trace2 = go.Scatter(x=[0, 1], y=[0, 1],
                        mode='lines',
                        line=dict(color='navy', width=lw, dash='dash'),
                        showlegend=False)

    layout = go.Layout(title='Receiver Operating Characteristic',
                       xaxis=dict(title='False Positive Rate'),
                       yaxis=dict(title='True Positive Rate'))
    gp={
        'data': [trace1, trace2],
        'layout': {
            'xaxis':{
                'title': 'False Positive Rate',
                'type': 'linear'
            },
            'yaxis':{
                'title': 'True Positive Rate',
                'type': 'linear'
            },
            'margin':{'l': 40, 'b': 40, 't': 10, 'r': 0},
            'hovermode':'closest',
        }
    }
    return gp

# Start Server
if __name__ == '__main__':
    app.run_server(debug=True)