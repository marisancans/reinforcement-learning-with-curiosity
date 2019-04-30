import pandas as pd 

import plotly.plotly as py
import plotly.graph_objs as go
import plotly
import numpy as np

from collections import defaultdict

plotly.tools.set_credentials_file(username='xonecell', api_key='6C5ZvbPGAQ9CkEqEs2GY')

import pandas as pd
import os
from scipy import signal

repeats = defaultdict(list)
data = []

dirs = [x[1] for x in os.walk('graphs/runs')][0]

for d in dirs:
    file_name = 'graphs/runs/' + d + '/' + d + '.csv'
    df = pd.read_csv(file_name, delimiter = ',')

    df = df[['repeat_id', 'score_avg', 'episode']]
    
    key = str(int(df.repeat_id[0]))
    repeats[key].append(df)


for k, v in repeats.items():
    all_score = []

    for df in v:
        all_score.append(df.score_avg.values)

    all_score = np.array(all_score)
    avg = np.average(all_score, axis=0)
    std = np.std(all_score, axis=0)

    name = 'Prioritized experience replay' if k=='1' else 'Randomly and uniformly'
    b = go.Scatter(
        name= name ,
        y=avg,
        mode='lines')
    
    data.append(b)

layout = go.Layout(
    yaxis=dict(title='Average score'),
    xaxis=dict(title='Episode', range=[1, len(all_score[0])],),
    title='PER vs random',
    showlegend = True)
    

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='PER vs random')


x = 1
