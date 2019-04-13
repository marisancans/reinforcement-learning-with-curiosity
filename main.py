# pip install box2d-py
import Box2D, os, shutil, datetime, time, random, sys, torch, argparse, gym
from gym import envs
import numpy as np
import torch.multiprocessing as mp
#from visualdl import LogWriter

from agent import Agent

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mode', type=int, default=1, help='0 - compare two models, 1 - run single model, 2 - DQN vs DDQN comparison, 3 - multiprocess testing')
parser.add_argument('-d', '--device', default='cpu', help='cpu or cuda')

parser.add_argument('-e', '--env_name', default='CartPole-v0',  help='OpenAI game enviroment name')
parser.add_argument('-lr', '--learning_rate', type=float, default=0.001)
parser.add_argument('-bs', '--batch_size', type=int, default=32)

parser.add_argument('--has_normalized_state', type=int, default=0, help='Normalize state vector in forward and inverse models? true | false')
parser.add_argument('--epsilon_decay', type=float, default=0.001, help='Epslion decay for exploration / explotation policy')
parser.add_argument('--epsilon_floor', type=float, default=0.01, help='Where epsilon stops to decay')
parser.add_argument('-g', '--gamma', type=float, default=0.95, help='Hyperparameter for DQN')
parser.add_argument('-ne', '--n_episodes', type=int, default=50, help='Number of episodes (games) to be played')
parser.add_argument('-nf', '--n_frames', type=int, default=9999, help='Number of frames per one episode')

parser.add_argument('--has_curiosity', type=int, default=0)
parser.add_argument('--beta_curiosity', type=float, default=-1, help='Beta hyperparameter for curiosity module')
parser.add_argument('--lambda_curiosity', type=float, default=-1, help='Lambda hyperparameter for curiosity module')
parser.add_argument('--curiosity_scale', type=float, default=1, help='Intrinsic reward scale factor')

parser.add_argument('--encoder_1_layer_out', type=int, default=5)
parser.add_argument('--encoder_2_layer_out', type=int, default=10)
parser.add_argument('--encoder_3_layer_out', type=int, default=15)

parser.add_argument('--inverse_1_layer_out', type=int, default=30)
parser.add_argument('--inverse_2_layer_out', type=int, default=20)

parser.add_argument('--forward_1_layer_out', type=int, default=30)
parser.add_argument('--forward_2_layer_out', type=int, default=20)

parser.add_argument('--has_ddqn', type=int, default=0, help='Is double DQN enabled?')
parser.add_argument('--target_update', type=float, default=10, help='Update target network after n steps')

args = parser.parse_args()

#logdir = "./logs"
#logger = LogWriter(logdir, sync_cycle=1000)

if torch.cuda.is_available():
    print('cuda detected')

mode = { 
        0: 'comparison',
        1: 'evaluate',
        2: 'dqn_vs_ddqn',
        3: 'multiprocess'
    }

def main():
    print('Running mode: {}'.format(mode[args.mode]))
    run = {
        0: comparison,
        1: evaluate,
        2: dqn_vs_ddqn,
        3: multiprocess,
        }
    
    run[args.mode]()

# This mode compares two agents
def comparison():
    with logger.mode('curious_ers'):
        logger_curious_ers = logger.scalar(mode[args.mode] + "/curious_ers")
    
    with logger.mode('dumb_ers'):
        logger_dumb_ers = logger.scalar(mode[args.mode] + "/dumb_ers")

    agent_curious = Agent(args)

    # change so that args are curious false
    agent_dumb = Agent(args)

    for i_episode in range(1, args.n_episodes):
        agent_curious.reset_env()
        agent_dumb.reset_env()

        start = time.time()
        is_done_c = False
        is_done_d = False

        while not is_done_c:
            is_done_c = agent_curious.play_step()

        while not is_done_d:
            is_done_d = agent_dumb.play_step()
        
        if not agent_curious.ers or not agent_dumb.ers:
            continue

        dqn = { 'c_dqn': agent_curious.loss_dqn[-1], 'd_dqn': agent_dumb.loss_dqn[-1] }
        writeDict('both_loss_dqn', dqn, i_episode)

        ers = { 'c_ers': agent_curious.ers[-1], 'd_ers': agent_dumb.ers[-1] }
        writeDict('aaa both_ers', ers, i_episode)


        logger_ers.add_record(i_episode, agent_curious.ers[-1])
        logger_ers_ddqn.add_record(i_episode, )

        #writeBoard('curious_loss_inverse', agent_curious.loss_inverse[-1], i_episode)                                 
        #writeBoard('curious_cos_distance', agent_curious.cos_distance[-1], i_episode)
        #writeBoard('curious_loss_combined', agent_curious.loss_combined[-1], i_episode) 


        t = (time.time() - start)*1000

        print("n: {}  |    epsilon: {:.2f}    |    ers_c:  {:.2f}    |    ers_d:  {:.2f}    |     time: {:.2f} s   ".format(
                i_episode, agent_curious.epsilon, agent_curious.ers[-1], agent_dumb.ers[-1], t*0.001))

# run just one agent
def evaluate():
   # with logger.mode('ers'):
        #logger_ers = logger.scalar(mode[args.mode] + "/ers")

    agent = Agent(args)
    start_all = time.time()
   
    for i_episode in range(1, args.n_episodes + 1):
        start = time.time()
        agent.reset_env()
        is_done = False
        
        while not is_done:
            is_done = agent.play_step()
            
        if not agent.ers:
            continue

        ers = agent.ers[-1]
        dqn_loss = agent.loss_dqn[-1]
        t = (time.time() - start)*1000

        if agent.args.has_curiosity:
            combined_loss = agent.loss_combined[-1]
            inverse_loss = agent.loss_inverse[-1]
            cos_distance = agent.cos_distance[-1]
            print("n: {}  |    epsilon: {:.2f}    |    dqn:  {:.2f}    |    ers:  {:.2f}    |    com: {:.2f}    |    inv: {:.2f}   cos: {:.2f}    |   time: {:.2f}".format(
                i_episode, agent.epsilon, dqn_loss, ers, combined_loss, inverse_loss, cos_distance, t))
        else:
            a = 1
            print("{}   |    n: {}  |    epsilon: {:.2f}    |    dqn_loss:  {:.2f}    |    ers:  {}    |     time: {:.2f}".format(
            args.device, i_episode, agent.epsilon, dqn_loss, ers, t))
        
        #logger_ers.add_record(i_episode, agent.ers[-1])

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