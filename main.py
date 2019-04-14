# pip install box2d-py
from visualdl import LogWriter # Has to be imported before everython elese, otherwise dumps core
import Box2D, os, shutil, datetime, time, random, sys, torch, argparse, gym, struct
from gym import envs
import numpy as np
import torch.multiprocessing as mp


# http://visualdl.paddlepaddle.org/documentation/visualdl/en/develop/getting_started/quick_start_en.html

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
parser.add_argument('--memory_size', type=int, default=10000, help="Replay memory size (This code uses sum tree, not deque)")
parser.add_argument('--parralel_runs', type=int, default=10, help="How many parralel agents to simulate")
#parser.add_argument('--state_min_val', type=int, default=-1, help)

parser.add_argument('--has_curiosity', type=int, default=0)
parser.add_argument('--curiosity_beta', type=float, default=-1, help='Beta hyperparameter for curiosity module')
parser.add_argument('--curiosity_lambda', type=float, default=-1, help='Lambda hyperparameter for curiosity module')
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

logdir = "./logs"
logW = LogWriter(logdir, sync_cycle=1000)

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


class TimeSlice():
    __slots__ = ["name", "i_episode", "ers_min", "ers_max", "ers_avg"]

    def __init__(self, agent, i_episode):
        self.name = agent.name
        self.i_episode = i_episode
        self.ers_min = min(agent.ers)
        self.ers_avg = sum(agent.ers) / len(agent.ers)
        self.ers_max = max(agent.ers)

def make_time_slice(agents, i_episode):    
    return [TimeSlice(a, i_episode) for a in agents]

# Log n parralel runs from timeline
def log_timeline(timelines, agent_names):
    for timeline, agent_name in zip(timelines, agent_names):
        with logW.mode(agent_name):
            logger_ers_min = logW.scalar("ers/min")
            logger_ers_avg = logW.scalar("ers/avg")
            logger_ers_max = logW.scalar("ers/max")

        for parralel_time_slices in timeline:
            parralel_min = min([time_slice.ers_min for time_slice in parralel_time_slices])
            parralel_max = max([time_slice.ers_max for time_slice in parralel_time_slices])
            parralel_avg = sum([time_slice.ers_avg for time_slice in parralel_time_slices]) / len(parralel_time_slices)

            logger_ers_min.add_record(parralel_time_slices[0].i_episode, parralel_min)
            logger_ers_avg.add_record(parralel_time_slices[0].i_episode, parralel_max)
            logger_ers_max.add_record(parralel_time_slices[0].i_episode, parralel_avg)

        


# This mode compares two agents
# args.parralel_runs is how many parralel agents are simulated and then taken average of
def comparison():
    # Parralel run timeline
    timeline_names = ['curious', 'dumb']
    timelines = [] 
     
    for run in range(args.parralel_runs):
        curious_args = args
        curious_args.has_curiosity = True
        agent_curious = Agent(curious_args, name='curious')

        agent_dumb = Agent(args, name='dumb')

        run_timeline = []

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
            
            e_ers_curious = agent_curious.ers[-1]  # ERS - Episode reward sum
            e_ers_dumb = agent_dumb.ers[-1]

            t = (time.time() - start)*1000

            print("run: {}   |   i_episode: {}  |    epsilon: {:.2f}    |    ers_c:  {:.2f}    |    ers_d:  {:.2f}    |     time: {:.2f} s   ".format(
                    run, i_episode, agent_curious.epsilon, e_ers_curious, e_ers_dumb, t*0.001))

            curious_slice, dumb_slice = make_time_slice(agents=[agent_curious, agent_dumb], i_episode=i_episode)
            timeline_slice = [curious_slice, dumb_slice]
            run_timeline.append(timeline_slice)

        timelines.append(run_timeline)
        # End of i_episodes
    # End of runs
    
    log_timeline(timelines, agent_names=timeline_names)




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
            
        ers = agent.ers[-1] if agent.ers else 0
        dqn_loss = agent.loss_dqn[-1] if agent.loss_dqn else 0

        t = (time.time() - start)*1000

        if agent.args.has_curiosity:
            loss_combined = agent.loss_combined[-1] if agent.loss_combined else 0
            loss_inverse = agent.loss_inverse[-1] if agent.loss_inverse else 0
            cos_distance = agent.cos_distance[-1] if agent.cos_distance else 0

            print("n: {}  |    epsilon: {:.2f}    |    dqn:  {:.2f}    |    ers:  {:.2f}    |    com: {:.2f}    |    inv: {:.2f}   cos: {:.2f}    |   time: {:.2f}".format(
                i_episode, agent.epsilon, dqn_loss, ers, loss_combined, loss_inverse, cos_distance, t))
        else:
            a = 1
            print("{}   |    n: {}  |    epsilon: {:.2f}    |    dqn_loss:  {:.2f}    |    ers:  {}    |     time: {:.2f}".format(
            args.device, i_episode, agent.epsilon, dqn_loss, ers, t))
        
        #logger_ers.add_record(i_episode, agent.ers[-1])

    print('finished in {:.2f} s'.format((time.time() - start_all)))

    
def dqn_vs_ddqn():
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