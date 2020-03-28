# -*- coding: utf-8 -*-
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html

from app import app
from DelayApps import DA1,DA2,DA3
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css','style.css']

print(dcc.__version__) # 0.6.0 or above is required

layout = html.Div([
    dcc.Location(id='url-3', refresh=False),
    html.Div(id='page-content-3')
])

index_page_3 =html.Div([
        html.Div( 
                 [html.H3([html.Span('DELAY DASHBOARDS',className="main-heading-primary")]
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
    
    html.Div([dcc.Link('DELAY PREDICTIONS', href='/DelayApps/DA1')],style={'display':'block','position' : 'absolute','margin-left':'50px','margin-top':'200px'}),
    html.Br(),
    html.Div([dcc.Link('DELAY PREDICTIONS WITH FEATURE CONTROL', href='/DelayApps/DA2')],style={'display':'block','position' : 'absolute','margin-left':'430px','margin-top':'175px'}),
    html.Br(),
    html.Div([dcc.Link('DELAY PREDICTIONS WITH FEATURE CONTROL ON O and D', href='/DelayApps/DA3')],style={'display':'block','position' : 'absolute','margin-left':'810px','margin-top':'150px'}),
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
@app.callback(Output('page-content-3', 'children'),
              [Input('url-3', 'pathname')])
def display_page(pathname_3):
    if(pathname_3=='/DelayApps/DA1'):
        print("in layout delay")
        return DA1.layout
    elif(pathname_3=='/DelayApps/DA2'):
        print("in layout delay")
        return DA2.layout
    elif(pathname_3=='/DelayApps/DA3'):
        print("in layout delay")
        return DA3.layout
    else:
        return index_page_3

if __name__ == '__main__':
    app.run_server(debug=True)


