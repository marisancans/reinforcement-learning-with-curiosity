import pandas as pd 
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
import seaborn as sns; sns.set()
import matplotlib

df = pd.read_csv('Acrobot-v1.csv', delimiter = ',')
df = df[['batch_size', 'curiosity_beta', 'curiosity_lambda', 'ers_avg']]
df = df.sort_values(by=['ers_avg'], ascending=False)
#df = df.head(25)
print(df)


data = df.to_numpy()
matrix = np.zeros(shape=(11, 11))

for batch_size, beta, lamda, ers in data:
    if batch_size != 32.0:
        continue

    b_idx = int(beta * 10)
    l_idx = int(lamda * 10)
    matrix[b_idx][l_idx] = 500 - abs(ers) 

minval = np.min(matrix[np.nonzero(matrix)])
maxval = np.max(matrix[np.nonzero(matrix)])


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
