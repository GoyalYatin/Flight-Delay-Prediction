# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 10:29:57 2019

@author: bsoni
"""
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html

from app import app

from EDA import edaindex
from Cancellations import cancellationindex
from Delays import delayindex
from EDAapps import edadelayindex
from EDAapps import edacancellationindex

from CancellationApps import CA1,CA2
from DelayApps import DA1,DA2,DA3
from EDAapps.edadelayapps import app1,app2,app3,app4,app5,app6,app7,app8
from EDAapps.edacancellationapps import capp1,capp2,capp3,capp4,capp5

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css','style.css']

#app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
print(dcc.__version__) # 0.6.0 or above is required

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

index_page =  html.Div([
         html.Div( 
                 [html.H3([html.Span('PREDICTIVE FLIGHT ANALYSIS',className="main-heading-primary")]
                 ,className='main-heading'),
				],style={'margin-top':'-20px'}),
html.Div([
    
    html.Div([dcc.Link('EDA DASHBOARDS', href='/EDA/edaindex')],style={'display':'block','margin-left':'50px','margin-top':'200px','position' : 'absolute'}),
    html.Br(),
    html.Div([dcc.Link('CANCELLATION DASHBOARDS', href='/Cancellations/cancellationindex')],style={'display':'block','margin-left':'430px','margin-top':'175px','position' : 'absolute'}),
    html.Br(),
    html.Div([dcc.Link('DELAY DASHBOARDS', href='/Delays/delayindex')],style={'display':'block','margin-left':'810px','margin-top':'150px','position' : 'absolute'}),
    html.Br(),
 
],),
 
],)

@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/EDA/edaindex':
         return edaindex.layout
    elif pathname == '/Cancellations/cancellationindex':
         print("in layout")
         return cancellationindex.layout
    elif pathname == '/Delays/delayindex':
         print("in layout delay")
         return delayindex.layout
    elif pathname == '/CancellationApps/CA1':
         print("in layout 2")
         return CA1.layout
    elif pathname == '/CancellationApps/CA2':
         print("in layout 2")
         return CA2.layout
    elif pathname == '/DelayApps/DA1':
         print("in layout 3")
         return DA1.layout
    elif pathname == '/DelayApps/DA2':
         print("in layout 4")
         return DA2.layout
    elif pathname == '/DelayApps/DA3':
         print("in layout 4")
         return DA3.layout
    elif pathname == '/EDAapps/edadelayindex':
         print("in layout 5")
         return edadelayindex.layout
    elif pathname == '/EDAapps/edacancellationindex':
         print("in layout cancel eda")
         return edacancellationindex.layout
    elif pathname == '/EDAapps/edadelayapps/app1':
         print("in layout 6")
         return app1.layout
    elif pathname == '/EDAapps/edadelayapps/app2':
         print("in layout 7")
         return app2.layout
    elif pathname == '/EDAapps/edadelayapps/app3':
         print("in layout 8")
         return app3.layout
    elif pathname == '/EDAapps/edadelayapps/app4':
         print("in layout 9")
         return app4.layout
    elif pathname == '/EDAapps/edadelayapps/app5':
         print("in layout 10")
         return app5.layout
    elif pathname == '/EDAapps/edadelayapps/app6':
         print("in layout 11")
         return app6.layout
    elif pathname == '/EDAapps/edadelayapps/app7':
         print("in layout 12")
         return app7.layout
    elif pathname == '/EDAapps/edadelayapps/app8':
         print("in layout 12")
         return app8.layout
    elif pathname == '/EDAapps/edacancellationapps/capp1':
         print("in layout 13")
         return capp1.layout
    elif pathname == '/EDAapps/edacancellationapps/capp2':
         print("in layout 14")
         return capp2.layout
    elif pathname == '/EDAapps/edacancellationapps/capp3':
         print("in layout 15")
         return capp3.layout
    elif pathname == '/EDAapps/edacancellationapps/capp4':
         print("in layout 16")
         return capp4.layout
    elif pathname == '/EDAapps/edacancellationapps/capp5':
         print("in layout 17")
         return capp5.layout
    elif pathname == '/':
         print("in layout 17")
         return index_page
    else:
         return index_page
    
   

if __name__ == '__main__':
    app.run_server(debug=True)
