# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 10:29:57 2019

@author: bsoni
"""
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
from app import app
from EDAapps.edadelayapps import app1,app2,app3,app4,app5,app6,app7,app8


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css','style.css']

#app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
print(dcc.__version__) # 0.6.0 or above is required

layout = html.Div([
    dcc.Location(id='url-4', refresh=False),
    html.Div(id='page-content-4')
])
print("in eda delay index")
index_page_4 = html.Div([html.Div( 
                 [html.H3([html.Span('EDA DELAY DASHBOARDS',className="main-heading-primary")]
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
    
    html.Div([dcc.Link('OTP dashboard: Airport Wise', href='/EDAapps/edadelayapps/app1')],style={'display':'block','position' : 'absolute','margin-left':'50px','margin-top':'200px'}),
    html.Br(),
    html.Div([dcc.Link('OTP Dashboard: Daytime Wise', href='/EDAapps/edadelayapps/app2')],style={'display':'block','position' : 'absolute','margin-left':'430px','margin-top':'175px'}),
    html.Br(),
    html.Div([dcc.Link('Taxi Out dashboard: Mean taxi-out time by airport', href='/EDAapps/edadelayapps/app3')],style={'display':'block','position' : 'absolute','margin-left':'810px','margin-top':'150px'}),
    html.Br(),
    html.Div([dcc.Link('Taxi Out dashboard: Taxi out time distributions', href='/EDAapps/edadelayapps/app4')],style={'display':'block','position' : 'absolute','margin-left':'1190px','margin-top':'125px'}),
    html.Br(),
    html.Div([dcc.Link('Taxi Out dashboard: Daytime Wise', href='/EDAapps/edadelayapps/app5')],style={'display':'block','position' : 'absolute','margin-left':'1570px','margin-top':'100px'}),
    html.Br(),
    html.Div([dcc.Link('Block time dashboard', href='/EDAapps/edadelayapps/app7')],style={'display':'block','position' : 'absolute','margin-left':'1950px','margin-top':'75px'}),
    html.Br(),
   ],),

 html.Div([   

    html.Div([dcc.Link('Delay Map', href='/EDAapps/edadelayapps/app8')],style={'display':'block','margin-left':'50px','margin-top':'400px','position' : 'absolute'}),
    html.Br(),
    
 
],),
 
])

@app.callback(Output('page-content-4', 'children'),
              [Input('url-4', 'pathname')])
def display_page(pathname_4):
    if pathname_4 == '/EDAapps/edadelayapps/app1':
         return app1.layout
    elif pathname_4 == '/EDAapps/edadelayapps/app2':
         return app2.layout
    elif pathname_4 == '/EDAapps/edadelayapps/app3':
         return app3.layout
    elif pathname_4 == '/EDAapps/edadelayapps/app4':
         return app4.layout
    elif pathname_4 == '/EDAapps/edadelayapps/app5':
         return app5.layout
    elif pathname_4 == '/EDAapps/edadelayapps/app7':
         return app7.layout
    else:
        return index_page_4

if __name__ == '__main__':
    app.run_server(debug=True)
