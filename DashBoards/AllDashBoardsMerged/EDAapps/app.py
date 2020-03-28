# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 10:28:06 2019

@author: bsoni
"""
import dash

app = dash.Dash()
server = app.server
app.config.supress_callback_exceptions = True

app.css.append_css({
    "external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"
})
