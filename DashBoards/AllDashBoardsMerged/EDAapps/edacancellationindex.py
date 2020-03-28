# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 10:29:57 2019

@author: bsoni
"""
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
from app import app
from EDAapps.edacancellationapps import capp1,capp2,capp3,capp4,capp5

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css','style.css']

#app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
print(dcc.__version__) # 0.6.0 or above is required

layout = html.Div([
    dcc.Location(id='url-5', refresh=False),
    html.Div(id='page-content-5')
])
print("in eda cancellation index")

index_page_5 = html.Div([html.Div( 
                 [html.H3([html.Span('EDA CANCELLATION DASHBOARDS',className="main-heading-primary")]
                 ,className='main-heading'),
				],style={'margin-top':'-20px'}),
html.Div([
                html.A([
                        html.Img(
                            src=app.get_asset_url('home.png'),
                            style={
                                'height' : '50px',
                                'width' : '50px',
                                'float' : 'right',
                                'position' : 'absolute',
                                'padding-top' : 0,
                                'padding-right' : 0,
                                'margin-left':'45',
                                'margin-top':'-40px',
                            })
                ],className='hello',href='/')
            ],style={'float':'top','margin-right':'1500px','height':'0px','width':'0px','background-color':'#ffffff'}) ,

html.Div([
                html.A([
                        html.Img(
                            src=app.get_asset_url('back.png'),
                            style={
                                'height' : '50px',
                                'width' : '50px',
                                'float' : 'right',
                                'position' : 'absolute',
                                'padding-top' : 0,
                                'padding-right' : 0,
                                'margin-right':'10px',
                                'margin-top':'-40px',
                            })
                ],className='hello',href='/EDA/edaindex')
            ],style={'float':'top','margin-left':'95%','height':'0px','width':'0px','background-color':'#ffffff'}) ,

html.Div([
    
    html.Div([dcc.Link('Cancellation Dashboard with Reason Distribution', href='/EDAapps/edacancellationapps/capp1')],style={'display':'block','position' : 'absolute','margin-left':'50px','margin-top':'200px'}),
    html.Br(),
    html.Div([dcc.Link('Cancellation Dashboard - Carrier wise', href='/EDAapps/edacancellationapps/capp2')],style={'display':'block','position' : 'absolute','margin-left':'430px','margin-top':'175px',}),
    html.Br(),
    html.Div([dcc.Link('Cancellation Dashboard - Origin-Destination wise', href='/EDAapps/edacancellationapps/capp3')],style={'display':'block','position' : 'absolute','margin-left':'810px','margin-top':'150px'}),
    html.Br(),
    html.Div([dcc.Link('Cancellation Dashboard - Airline Route wise', href='/EDAapps/edacancellationapps/capp4')],style={'display':'block','position' : 'absolute','margin-left':'1190px','margin-top':'125px'}),
    html.Br(),
    html.Div([dcc.Link('Cancellation Dashboard - Traffic against Cancellation', href='/EDAapps/edacancellationapps/capp5')],style={'display':'block','position' : 'absolute','margin-left':'1570px','margin-top':'100px'}),
    html.Br(),
    
 
],),
 
])

@app.callback(Output('page-content-5', 'children'),
              [Input('url-5', 'pathname')])
def display_page(pathname_5):
    if pathname_5 == '/EDAapps/edacancellationapps/capp1':
         return capp1.layout
    elif pathname_5 == '/EDAapps/edacancellationapps/capp2':
         return capp2.layout
    elif pathname_5 == '/EDAapps/edacancellationapps/capp3':
         return capp3.layout
    elif pathname_5 == '/EDAapps/edacancellationapps/capp4':
         return capp4.layout
    elif pathname_5 == '/EDAapps/edacancellationapps/capp5':
         return capp5.layout
    else:
        return index_page_5

if __name__ == '__main__':
    app.run_server(debug=True)
