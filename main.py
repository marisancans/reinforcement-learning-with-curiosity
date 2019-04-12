# # Box2d
# git clone https://github.com/pybox2d/pybox2d
# cd pybox2d/
# python setup.py clean
# python setup.py build
# python setup.py install
# pip install box2d-py  | bindings to library
import Box2D
import gym # openAi gym
from gym import envs
import numpy as np
import torch
import os, shutil, datetime, time, random, sys
import tensorboardX
import torch.multiprocessing as mp
import argparse
from visualdl import LogWriter

from agent import Agent

# this somehow fixes multiprocessing cuda error
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]=""

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mode', type=int)
parser.add_argument('-d', '--device')
args = parser.parse_args()

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
if torch.cuda.is_available():
    print('cuda detected')
    
device = "cpu"
env_name = 'CartPole-v0'
batch_size = 32
mode = args.mode

folder = {
        0: 'comparison',
        1: 'evaluate',
        2: 'dqn_vs_ddqn',
        3: 'multiprocess'
    }
print('Running mode: {}'.format(folder[mode]))
path_tf_log = 'tf_log/' + folder[mode] + "/" + env_name + "_" + str(batch_size)

if os.path.exists(path_tf_log):
    shutil.rmtree(path_tf_log)
if not os.path.exists(path_tf_log):
    os.makedirs(path_tf_log)

tensor_board_writer = tensorboardX.SummaryWriter(log_dir=path_tf_log)

def main():
    run = {
        0: comparison,
        1: evaluate,
        2: dqn_vs_ddqn,
        3: multiprocess,
        }

    run[mode]()


def writeBoard(tag, scalar, step):
    tensor_board_writer.add_scalar(tag=tag, scalar_value=scalar, global_step=step)

def writeDict(tag, dict_scalars, step):
    tensor_board_writer.add_scalars(main_tag=tag, tag_scalar_dict=dict_scalars, global_step=step)



# This mode compares agents
def comparison():
    n_episodes = 301
    decay = 0.0001
    agent_curious = Agent(device, env_name, batch_size=batch_size, curiosity=True, ddqn=False, beta=0.2, lamda=0.8, epsilon_decay=decay)
    agent_dumb = Agent(device, env_name, batch_size=batch_size, curiosity=False, ddqn=False, epsilon_decay=decay)

    for i_episode in range(1, n_episodes):
        agent_curious.reset_env()
        agent_dumb.reset_env()

        start = time.time()
        is_done_c = False
        is_done_d = False

        while not is_done_c:
            is_done_c = agent_curious.play_step()

        while not is_done_d:
            is_done_d = agent_dumb.play_step()

        dqn = { 'c_dqn': agent_curious.loss_dqn[-1], 'd_dqn': agent_dumb.loss_dqn[-1] }
        writeDict('both_loss_dqn', dqn, i_episode)

        ers = { 'c_ers': agent_curious.ers[-1], 'd_ers': agent_dumb.ers[-1] }
        writeDict('aaa both_ers', ers, i_episode)





        writeBoard('curious_loss_inverse', agent_curious.loss_inverse[-1], i_episode)                                 
        writeBoard('curious_cos_distance', agent_curious.cos_distance[-1], i_episode)
        writeBoard('curious_loss_combined', agent_curious.loss_combined[-1], i_episode) 


        t = (time.time() - start)*1000

        print("n: {}  |    epsilon: {:.2f}    |    ers_c:  {:.2f}    |    ers_d:  {:.2f}    |     time: {:.2f} s   ".format(
                i_episode, agent_curious.epsilon, agent_curious.ers[-1], agent_dumb.ers[-1], t*0.001))

# run just one agent
def evaluate():
    logdir = "./tmp"
    logger = LogWriter(logdir, sync_cycle=10)
    with logger.mode("train"):
        logger_ers = logger.scalar("scalars/ers")


    env_name='MountainCar-v0'
    device = args.device
    decay = 0.00002
    agent = Agent(device, env_name, batch_size=16, epsilon_decay=decay, beta=0.8, lamda=0.2, curiosity=True, ddqn=True, lr=0.001)
    agent.lr
    n_episodes = 500
    start_all = time.time()

    for i_episode in range(1, n_episodes):
        start = time.time()
        agent.reset_env()
        is_done = False
        while not is_done:
            is_done = agent.play_step()
            #if any(x > -200 for x in agent.ers):
             #   agent.env.render()

        ers = agent.ers[-1]
        dqn_loss = agent.loss_dqn[-1]
        t = (time.time() - start)*1000

        if agent.curiosity:
            combined_loss = agent.loss_combined[-1]
            inverse_loss = agent.loss_inverse[-1]
            cos_distance = agent.cos_distance[-1]
            print("n: {}  |    epsilon: {:.2f}    |    dqn:  {:.2f}    |    ers:  {:.2f}    |    com: {:.2f}    |    inv: {:.2f}   cos: {:.2f}    |   time: {:.2f}".format(
                i_episode, agent.epsilon, dqn_loss, ers, combined_loss, inverse_loss, cos_distance, t))
        else:
            a = 1
            print("{}   |    n: {}  |    epsilon: {:.2f}    |    dqn_loss:  {:.2f}    |    ers:  {}    |     time: {:.2f}".format(
            device, i_episode, agent.epsilon, dqn_loss, ers, t))
        writeBoard('loss_dqn', agent.loss_dqn[-1], i_episode)
        writeBoard('ers', agent.ers[-1], i_episode)
        
        logger_ers.add_record(i_episode, agent.ers[-1])

    print('finished in {:.2f} s'.format((time.time() - start_all)))

    
def dqn_vs_ddqn():
    n_episodes = 50
    decay = 0.0001
    update = 25
    env_name = 'Acrobot-v1'

    logdir = "./tmp"
    logw = LogWriter(logdir, sync_cycle=1000)


    with logw.mode('dqn') as logger:
        logger_dqn = logger.scalar("scalar")

    with logw.mode('ddqn') as logger:
        logger_ddqn = logger.scalar("scalar")

    for game in range(10):
        agent = Agent(device, env_name, batch_size=batch_size, curiosity=False, ddqn=False, epsilon_decay=decay)
        agent_ddqn = Agent(device, env_name, batch_size=batch_size, curiosity=False, ddqn=True, epsilon_decay=decay, update_target_every=update)


        for i_episode in range(1, n_episodes):
            agent.reset_env()
            agent_ddqn.reset_env()

            start = time.time()
            is_done = False
            is_done_ddqn = False

            while not is_done:
                is_done = agent.play_step()

            while not is_done_ddqn:
                is_done_ddqn = agent_ddqn.play_step()

            ers = { 'ers': agent.ers[-1], 'ers_ddqn': agent_ddqn.ers[-1] }
            writeDict('ers update: ' + str(agent_ddqn.update_target_every) + ' decay: ' + str(agent_ddqn.epsilon_decay), ers, i_episode)

            dqn = { 'dqn': agent.loss_dqn[-1], 'dqn_ddqn': agent_ddqn.loss_dqn[-1] }
            writeDict('aaaaboth_loss_dqn', dqn, i_episode)

            t = (time.time() - start)*1000

            print("n: {}  |    epsilon: {:.2f}    |    ers:  {:.2f}    |    ers_ddqn:  {:.2f}    |     time: {:.2f} s    ".format(
                    i_episode, agent.epsilon, agent.ers[-1], agent_ddqn.ers[-1], t*0.001))

        ers_dqn_avg = sum(agent.ers) / len(agent.ers)
        ers_ddqn_avg = sum(agent_ddqn.ers) / len(agent_ddqn.ers)
    
        logger_dqn.add_record(game, ers_dqn_avg)
        logger_ddqn.add_record(game, ers_ddqn_avg)

    
def multiprocess():
    processes = []
    for i in range(1):
        print(i)
        p = mp.Process(target=evaluate, args=())
        processes.append(p)
        p.start()
    for p in processes:
        p.join()

if __name__ == '__main__':
    main()