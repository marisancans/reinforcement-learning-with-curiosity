import plotly.plotly as py
import plotly.graph_objs as go
import plotly

plotly.tools.set_credentials_file(username='xonecell', api_key='6C5ZvbPGAQ9CkEqEs2GY')

import pandas as pd
import os
from scipy import signal

file_name = 'graphs/ers_data.csv'

df = pd.read_csv(os.path.join(os.getcwd(), file_name), delimiter = ',')

agents = []
data = []

agent_names = ['dqn', 'ddqn', 'curious', 'curious_ddqn']

fillcolors = {
    '0': 'rgba(255, 191, 0, 0.1)',
    '1': 'rgba(110, 255, 0 0.1)',
    '2': 'rgba(255, 0, 0, 0.1)',
    '3': 'rgba(0, 0, 255, 0.1)',
}

linecolors = {
    '0': 'rgba(255, 191, 0, 1)',
    '1': 'rgba(110, 255, 0 1)',
    '2': 'rgba(255, 0, 0, 1)',
    '3': 'rgba(0, 0, 255, 1)',
}

def smooth_criminal(x):
    return signal.savgol_filter(x, 53, 3)

for a in agent_names:
    agents.append({
        'name': a,
        a + '_avg_score': df[a + '_avg_score'].values,
        a + '_std': df[a + '_std'].values,
        }) 

for idx, a in enumerate(agents):
    name = a['name']

    y = smooth_criminal(df[name + '_avg_score']+df[name + '_std'])
    upper_bound = go.Scatter(
        name= name + ' upper',
        y=y,
        mode='lines',
        marker=dict(color="#444"),
        line=dict(width=0),
        fillcolor=fillcolors[str(idx)],
        fill='tonexty')

    y = smooth_criminal(df[name + '_avg_score'])
    trace = go.Scatter(
        name= name + ' avg score',
        y=y,
        mode='lines',
        line=dict(color=linecolors[str(idx)]),
        fillcolor=fillcolors[str(idx)],
        fill='tonexty')

    y = smooth_criminal(df[name + '_avg_score']-df[name + '_std'])
    lower_bound = go.Scatter(
        name= name + ' lower',
        y=y,
        marker=dict(color="#444"),
        line=dict(width=0),
        mode='lines')

    # Trace order can be important
    # with continuous error bars
    
    data.append(lower_bound)
    data.append(trace)
    data.append(upper_bound)


layout = go.Layout(
    yaxis=dict(title='Score'),
    xaxis=dict(title='Episode'),
    title='Continuous, variable value error bars.<br>Notice the hover text!',
    showlegend = False)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='pandas-continuous-error-bars')
