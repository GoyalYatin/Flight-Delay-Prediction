# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 21:15:31 2019

@author: bsoni
"""
import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import numpy as np
import datetime
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import cmocean
import matplotlib.pyplot as plt
import matplotlib
import squarify
from app import app

df = pd.read_csv("us-bts_parsed_data_delay_reason.csv")
dfdelay = pd.read_csv("us-bts_parsed_data_delay_reason.csv")
flightsdelay = dfdelay.carrier.unique()

#df.drop(['Unnamed: 0'], axis=1,inplace =True)
#df = df.dropna()
flights = df.carrier.unique()
j=[1]
print("this print every time")
flight_origin = df.origin.unique()
layout = html.Div([
    html.Div([
                html.Div([
    
    html.Div([dcc.Link('OTP dashboard: by airport', href='/page-1')],style={'padding': '6px 8px 6px 16px',
  'text-decoration': 'none',
  'font-size': '12px',
  'color': '#818181',
  'display':'block',}),
    html.Br(),
    html.Div([dcc.Link('OTP Dashboard: by time of the day', href='/page-2')],style={'padding': '6px 8px 6px 16px',
  'text-decoration': 'none',
  'font-size': '12px',
  'color': '#818181',
  'display':'block',}),
    html.Br(),
    html.Div([dcc.Link('Taxi Out dashboard: Mean taxi-out time by airport', href='/page-3')],style={'padding': '6px 8px 6px 16px',
  'text-decoration': 'none',
  'font-size': '12px',
  'color': '#818181',
  'display':'block',}),
    html.Br(),
    html.Div([dcc.Link('Taxi Out dashboard: Taxi out time distributions', href='/page-4')],style={'padding': '6px 8px 6px 16px',
  'text-decoration': 'none',
  'font-size': '12px',
  'color': '#818181',
  'display':'block',}),
    html.Br(),
    html.Div([dcc.Link('Taxi Out dashboard: By time of day', href='/page-5')],style={'padding': '6px 8px 6px 16px',
  'text-decoration': 'none',
  'font-size': '12px',
  'color': '#818181',
  'display':'block',}),
    html.Br(),
    html.Div([dcc.Link('Delay reason dashboard', href='/page-6')],style={'padding': '6px 8px 6px 16px',
  'text-decoration': 'none',
  'font-size': '12px',
  'color': '#818181',
  'display':'block',}),
    html.Br(),
    html.Div([dcc.Link('Block time dashboard', href='/apps/app7')],style={'padding': '6px 8px 6px 16px',
  'text-decoration': 'none',
  'font-size': '12px',
  'color': '#818181',
  'display':'block',}),
  html.Br(),
 html.Div([dcc.Link('Delay trends', href='/apps/app8')],style={'padding': '6px 8px 6px 16px',
  'text-decoration': 'none',
  'font-size': '12px',
  'color': '#818181',
  'display':'block',}),
html.Br(),
 html.Div([dcc.Link('Cancellation trends', href='/apps/app9')],style={'padding': '6px 8px 6px 16px',
  'text-decoration': 'none',
  'font-size': '12px',
  'color': '#818181',
  'display':'block',}),
],
style={
 
  'height': '100%',   'width': '200px', 
  'position': 'fixed',
  'z-index': '1',   'top': '0',
  'left': '0',
  'background-color': '#111',
  'overflow-x': 'hidden',
  'padding-top': '20px',
}),


        html.Div([html.P('Select a carrier :'),dcc.Dropdown(
                    id='xaxis-column-5',
                    options=[{'label': i, 'value': i} for i in flightsdelay],
                    value='WN'
            ),
            
        ],
        style={'width': '48%', 'display': 'inline-block','margin-left': '220px'}),


        
         html.Div([
    dcc.Graph(id='treemap'),
    
        ],style={'width': '45%', 'display': 'inline-block', 'padding': '0 20','margin-left': '220px'}),
                    
       
    ]),

])


@app.callback(
    dash.dependencies.Output('treemap', 'figure'),
    [dash.dependencies.Input('xaxis-column-5', 'value'),
     dash.dependencies.Input('treemap', 'clickData'),
     ])

def update_graph(xaxis_column,clickData):
    clickCounter=1
    dff = dfdelay[dfdelay['carrier'] == xaxis_column]
    B = []
    Dd = []
    K= []
    myColors=[]
    List = dff.origin.unique()
    for i in List:
        dftemp = dff[dff['origin']==i]
        Dd.append(i)
        B.append(dftemp.shape[0])
        K.append(dftemp['dep_delay'].mean())
    
    cmap = matplotlib.cm.Reds
    mini=min(K)
    maxi=max(K)
    x = 0.
    y = 0.
    width = 100.
    height = 100.
    normed = squarify.normalize_sizes(B, width, height)
    rects = squarify.squarify(normed, x, y, width, height)
    shapes = []
    annotations = []

    counter = 0
    norm = matplotlib.colors.Normalize(vmin=mini, vmax=maxi)
    colors = [cmap(norm(value)) for value in K]
    for i in colors:
        a = str(int(round(i[0]*100)))
        b = str(int(round(i[1]*100)))
        c = str(int(round(i[2]*100)))
        m = 'rgb('+a+','+b+','+c+')'
        myColors.append(m)
    
    toShow=[]
    for i in range(len(B)):
        str1 = str(B[i])
        str2 = str(int(K[i]))
        toShow.append('No. of flights:'+str1+'\n'+'Mean dep delay:'+str2)
        
    print(toShow)
    shapes = []
    annotations = []
    counter = 0

    for r in rects:
        shapes.append( 
            dict(
                type = 'rect', 
                x0 = r['x'], 
                y0 = r['y'], 
                x1 = r['x']+r['dx'], 
                y1 = r['y']+r['dy'],
                line = dict( width = 2 ),
                fillcolor = myColors[counter]
            ) 
        )
        annotations.append(
            dict(
                x = r['x']+(r['dx']/2),
                y = r['y']+(r['dy']/2),
                text = Dd[counter],
                showarrow = False,
                opacity = 1,
                font = dict(
                        color = "black",
                        size = 12
                )
            )
        )
        counter = counter + 1
   
        if counter >= len(myColors):
            counter = 0
    
    print("click counter in graph 1", clickCounter)
    figure = {
    'data': [go.Scatter(
        x = [ r['x']+(r['dx']/2) for r in rects ], 
        y = [ r['y']+(r['dy']/2) for r in rects ],
        text = [ str(v) for v in toShow ], 
        #text1 = [ str(v) for v in Dd ], 
        mode = 'text',
        textfont = dict(
                        color = "#ffffff",
                        size = 12
                ),
        fillcolor = '#ffffff',
        hoverinfo = 'text'
        
        )
    ],
    'layout': go.Layout(
            title = 'Delay statistics according to airports',
        height=900, 
        width=1300,
        xaxis={'showgrid':False, 'zeroline':False, 'showticklabels': False},
        yaxis={'showgrid':False, 'zeroline':False, 'showticklabels': False},
        shapes=shapes,
        annotations=annotations,
        hovermode='closest',
        
        )
    }
    if clickData is not None:
        no = clickData['points'][0]['pointNumber']
        st = Dd[no]
        print(st)
        dff = dff[dff['origin']==str(Dd[no])]
        print(dff.shape)
        ss = []
        dc1 = dff[dff['delay_reason']=='carrier_delay']
        dc2 = dff[dff['delay_reason']=='weather_delay']
        dc3 = dff[dff['delay_reason']=='nas_delay']
        dc4 = dff[dff['delay_reason']=='security_delay']
        dc5 = dff[dff['delay_reason']=='late_aircraft_delay']
        ss.append(dc1.shape[0])
        ss.append(dc2.shape[0])
        ss.append(dc3.shape[0])
        ss.append(dc4.shape[0])
        ss.append(dc5.shape[0])
        print('ss is')
        print(ss)
        qq=[]
        qq.append(dff['carrier_delay'].mean())
        qq.append(dff['weather_delay'].mean())
        qq.append(dff['nas_delay'].mean())
        qq.append(dff['security_delay'].mean())
        qq.append(dff['late_aircraft_delay'].mean())
        print('qq is')
        print(qq)
        zs=[]
        count=0
        for i in qq:
            if (i==0):
                zs.append(count)
                count=count+1
            else:
                count=count+1
        
        print(zs)
        rr = ['Carrier Delay','Weather Delay','NAS Delay','Security Delay','Late Aircraft Delay']
        qq = [i for j, i in enumerate(qq) if j not in zs]
        ss = [i for j, i in enumerate(ss) if j not in zs]
        rr = [i for j, i in enumerate(rr) if j not in zs]
        print(rr)
        print(qq)
        print(ss)
        toShowSmallText = []
        for i in range(len(ss)):
            str11 = str(ss[i])
            str12 = str(qq[i])
            toShowSmallText.append("mean reason Delay:"+str12+"  reason delay count:"+str11)
        x = 0.
        y = 0.
        width = 100.
        height = 100.
        normed = squarify.normalize_sizes(qq, width, height)
        rects1 = squarify.squarify(normed, x, y, width, height)
        cmap = matplotlib.cm.Reds
        mini=min(ss)
        maxi=max(ss)
        norm = matplotlib.colors.Normalize(vmin=mini, vmax=maxi)
        colors = [cmap(norm(value)) for value in ss]
        for i in colors:
            a = str(int(round(i[0]*100)))
            b = str(int(round(i[1]*100)))
            c = str(int(round(i[2]*100)))
            m = 'rgb('+a+','+b+','+c+')'
            myColors.append(m)
            shapes = []
        annotations = []
        counter = 0
        for r in rects1:
            shapes.append( 
                    dict(
                            type = 'rect', 
                            x0 = r['x'], 
                            y0 = r['y'], 
                            x1 = r['x']+r['dx'], 
                            y1 = r['y']+r['dy'],
                            line = dict( width = 2 ),
                            fillcolor = myColors[counter]
                            ) 
                    )
            annotations.append(
                    dict(
                            x = r['x']+(r['dx']/2),
                            y = r['y']+(r['dy']/2),
                            text = rr[counter],
                            showarrow = False,
                            opacity = 1,
                            font = dict(
                                    color = 'black',
                                    size = 12
                                    )
                            )
                            )
            counter = counter + 1
            if counter >= len(myColors):
                counter = 0

        clickCounter = clickCounter+1
        print("click counter in graph 2", clickCounter)
        print(figure['data'][0])
        figure1 = {
                'data': [go.Scatter(
                        x = [ r['x']+(r['dx']/2) for r in rects1 ], 
                        y = [ r['y']+(r['dy']/2) for r in rects1],
                        textfont = dict(
                        color = "#ffffff",
                        size = 12
                            ),
                        fillcolor = '#ffffff',
                        text = [ str(v) for v in toShowSmallText ], 
                        #text1 = [ str(v) for v in Dd ], 
                        mode = 'text',
                        hoverinfo = 'text'
                        
        
                        )
                    ],
            'layout': go.Layout(
                    title = 'Delay reason statistics',
                    height=700, 
                    width=700,
                    xaxis={'showgrid':False, 'zeroline':False, 'showticklabels': False},
                    yaxis={'showgrid':False, 'zeroline':False, 'showticklabels': False},
                    shapes=shapes,
                    annotations=annotations,
                    hovermode='closest',
        
                        )
                }
        
        print(Dd)
    if(clickCounter==0):
            return figure
    elif(clickCounter==2):
            return figure1
    elif(clickCounter==1):
            return figure
        
if __name__ == '__main__':
    app.run_server(debug=True)