import pandas as pd 
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
import seaborn as sns; sns.set()
import matplotlib

df = pd.read_csv('reports/mountain_car.csv', delimiter = ',')
df = df[['curiosity_beta', 'curiosity_lambda', 'score_avg']]
df = df.sort_values(by=['score_avg'], ascending=False)
#df = df.head(25)
print(df)


grid_size = 10 # 5 or 10

data = df.to_numpy()
matrix = np.zeros(shape=(grid_size + 1, grid_size + 1))

for  beta, lamda, score in data:
    b_idx = int(beta * grid_size)
    l_idx = int(lamda * grid_size)
    matrix[b_idx][l_idx] = 200 - abs(score)

minval = np.min(matrix[np.nonzero(matrix)])
maxval = np.max(matrix[np.nonzero(matrix)])

matrix = np.round(matrix, decimals=2)

ax = sns.heatmap(matrix, fmt='g', vmin=minval, vmax=maxval, cmap="plasma", annot=True,)
x_labels = np.round(np.arange(0, 1.1, 0.1), decimals=1)
y_labels = np.round(np.arange(1, -0.1, -0.1), decimals=1)
ax.set_xticklabels(labels=x_labels)
ax.set_yticklabels(labels=x_labels)
plt.draw()
plt.xlabel('lambda')
plt.ylabel('beta')
plt.show()
x = 1
