import pandas as pd 

import plotly.plotly as py
import plotly.graph_objs as go
import plotly

plotly.tools.set_credentials_file(username='xonecell', api_key='6C5ZvbPGAQ9CkEqEs2GY')

import pandas as pd
import os
from scipy import signal

file_name = 'graphs/abc.csv'
df = pd.read_csv(file_name, delimiter = ',', encoding='utf16')

df = df[['repeat_id', 'batch_size', 'learning_rate', 'epsilon_decay', 'score_avg']]



x = 1
