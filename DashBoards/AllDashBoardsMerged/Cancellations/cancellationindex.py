# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 10:29:57 2019

@author: bsoni
"""
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html

from app import app
#from apps import app1,app2,app3,app4,app5,app6,app7,app8,app9
from CancellationApps import CA1,CA2

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css','style.css']

#app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
print(dcc.__version__) # 0.6.0 or above is required

layout = html.Div([
    dcc.Location(id='url-2', refresh=False),
    html.Div(id='page-content-2')
])

index_page_2 =html.Div([
        html.Div( 
                 [html.H3([html.Span('CANCELLATION DASHBOARDS',className="main-heading-primary")]
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
    
    html.Div([dcc.Link('CANCELLATION PREDICTIONS', href='/CancellationApps/CA1')],style={'display':'block','position' : 'absolute','margin-left':'50px','margin-top':'200px'}),
    html.Br(),
    html.Div([dcc.Link('CANCELLATION PREDICTIONS WITH FEATURE CONTROL', href='/CancellationApps/CA2')],style={'display':'block','position' : 'absolute','margin-left':'430px','margin-top':'175px'}),
    html.Br(),
   
 
],),


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
@app.callback(Output('page-content-2', 'children'),
              [Input('url-2', 'pathname')])
def display_page(pathname_2):
    if(pathname_2=='/CancellationApps/CA1'):
        print("in layout cancel")
        return CA1.layout
    if(pathname_2=='/CancellationApps/CA2'):
        print("in layout cancel")
        return CA2.layout
    else:
        return index_page_2

if __name__ == '__main__':
    app.run_server(debug=True)


