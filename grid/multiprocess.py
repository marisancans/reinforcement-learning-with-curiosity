import os, time                                                                       
import multiprocessing    
from multiprocessing import Pool, Process                                           
from agent import Agent
import numpy as np  
from sklearn.model_selection import ParameterGrid
import pandas as pd
from multiprocessing import Process, Queue, Lock
from time import sleep

cpus = 3
print('cpu cunt: {}'.format(cpus))
    
def run(p):
    print(p)        
    
def writefile(df, file_name, lock):
    with lock:
        df.to_csv(fn, index=False, mode='a', header=False)

def init_child(lock_):
    global lock
    lock = lock_

def main():
    alpha = np.round(np.arange(0, 1.1, 0.1), 1)
    lamda = np.round(np.arange(0, 1.1, 0.1), 1)
    batch_size = [32]#[32, 64, 128, 256]
    param_grid = {'beta': alpha, 'lamda': lamda, 'batch_size': batch_size}
    env_name = ['CartPole-v0']
    p = ParameterGrid(param_grid)
    p = list(p)

    lock = Lock()
    pool = Pool(processes=cpus, initializer=init_child, initargs=(lock,))                                                        
    pool.map(run, p) 

if __name__ == "__main__":
    main()

