# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 10:29:57 2019

@author: bsoni
"""
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
from EDAapps import edadelayindex
from EDAapps import edacancellationindex
from app import app
#from apps import app1,app2,app3,app4,app5,app6,app7,app8,app9


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css','style.css']

#app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
print(dcc.__version__) # 0.6.0 or above is required

layout = html.Div([
    dcc.Location(id='url-1', refresh=False),
    html.Div(id='page-content-1')
])

index_page_1 = html.Div([
        html.Div( 
                 [html.H3([html.Span('EDA DASHBOARDS',className="main-heading-primary")]
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
    
    html.Div([dcc.Link('DELAY EDA DASHBOARDS', href='/EDAapps/edadelayindex')],style={'display':'block','margin-left':'50px','margin-top':'200px','position' : 'absolute'}),
    html.Br(),
    html.Div([dcc.Link('CANCELLATION EDA DASHBOARDS', href='/EDAapps/edacancellationindex')],style={'display':'block','position' : 'absolute','margin-left':'430px','margin-top':'175px'}),
    html.Br(),]),
   
])

"""

@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/apps/app1':
         return app1.layout
    elif pathname == '/apps/app2':
         return app2.layout
    elif pathname == '/apps/app3':
         return app3.layout
    elif pathname == '/apps/app4':
         return app4.layout
    elif pathname == '/apps/app5':
         return app5.layout
    elif pathname == '/apps/app6':
         return app6.layout
    elif pathname == '/apps/app7':
         return app7.layout
    elif pathname == '/apps/app8':
         return app8.layout
    elif pathname == '/apps/app9':
         return app9.layout
    else:
        return index_page
"""
#==============change pathname id, pagecontent id, url id
@app.callback(Output('page-content-1', 'children'),
              [Input('url-1', 'pathname')])
def display_page(pathname_1):
    if(pathname_1=='/EDAapps/edadelayindex'):
        return edadelayindex.layout
    elif(pathname_1=='/EDAapps/edacancellationindex'):
        return edacancellationindex.layout
    else:
        return index_page_1

if __name__ == '__main__':
    app.run_server(debug=True)


