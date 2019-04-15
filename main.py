# pip install box2d-py
from visualdl import LogWriter # Has to be imported before everython elese, otherwise dumps
import Box2D, os, shutil, datetime, time, random, sys, torch, argparse, gym, pandas
from gym import envs
import numpy as np
import multiprocessing
from sklearn.model_selection import ParameterGrid
from multiprocessing import Pool, Process, Lock
import pandas as pd

from agent import Agent


parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mode', type=int, default=1, help='0 - run single model, 1 - compare multiple agents , 2 - multiprocess testing')
parser.add_argument('-d', '--device', default='cpu', help='cpu or cuda')
parser.add_argument('--debug', type=int, default=0, help='Extra print statements between episodes')

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
parser.add_argument('--parralel_runs', type=int, default=5, help="How many parralel agents to simulate")
parser.add_argument('--n_processes', type=int, default=3, help="How many parralel processes to run (MODE 3)")
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

logdir = "./logs/"

d = logdir + 'comp'
logComp = LogWriter(d, sync_cycle=100)

if torch.cuda.is_available():
    print('cuda detected')

mode = { 
        0: 'comparison',
        1: 'evaluate',
        2: 'multiprocess'
    }

def main():
    print('Running mode: {}'.format(mode[args.mode]))
    run = {
        0: comparison,
        1: evaluate,
        2: multiprocess,
        }
    
    run[args.mode]()

# Log n parralel runs from timeline
def log_seperate_agent(ers_avg, agent_name):
    with logComp.mode(agent_name) as l:
        logger_ers_avg = l.scalar("/ers/avg")

    for t, a in enumerate(ers_avg):
        logger_ers_avg.add_record(t, a)




def log_parralel_agents(all_ers, prefix):
    # d = logdir + prefix
    # logW = LogWriter(d, sync_cycle=10000)

    # with logW.mode("ers_min"):
    #     logger_ers_min = logW.scalar(agent_name)

    # with logW.mode("ers_avg"):
    #     logger_ers_avg = logW.scalar(agent_name)

    # with logW.mode("ers_max"):
    #     logger_ers_max = logW.scalar(agent_name)


    ers_min = []
    ers_avg = []
    ers_max = []

    for a in agents:
        t_ers_avg = []

        for time_slice in t:
            t_ers_avg.append(time_slice.ers_avg)
            i_episode = time_slice.i_episode

        ers_avg.append(t_ers_avg)


    ers_avg = np.array(ers_avg)

    for x in range(ers_avg.shape[1]):
        timeslice_min = np.min(ers_avg[:, x], axis=0) 
        timeslice_avg = np.sum(ers_avg[:, x], axis=0) / len(ers_avg)
        timeslice_max = np.max(ers_avg[:, x], axis=0)
        
        logger_ers_min.add_record(x, timeslice_min)
        logger_ers_avg.add_record(x, timeslice_avg)
        logger_ers_max.add_record(x, timeslice_max)
        
        
   


# This mode compares n agents
# ==== MODE 0 ======
def comparison():
    # Parralel run timeline
    names = ['curious', 'curious_ddqn', 'dqn', 'ddqn']

    curious_args = args
    curious_args.has_curiosity = 1

    curious_ddqn_args = args
    curious_ddqn_args.has_curiosity = 1
    curious_ddqn_args.has_ddqn = 1

    ddqn_args = args
    ddqn_args.has_ddqn = 1

    all_args = [curious_args, curious_ddqn_args, args, ddqn_args]
    all_ers = np.zeros(shape=(args.parralel_runs, len(names), args.n_episodes))
   
    for run in range(args.parralel_runs):
        agents = [Agent(agent_args, name=n) for agent_args, n in zip(all_args, names)]

        for a_idx, a in enumerate(agents):
            for i_episode in range(args.n_episodes):
                a.reset_env()
                is_done = False

                while not is_done:
                    is_done = a.play_step()
                
                all_ers[run][a_idx][i_episode] = np.array(a.ers[-1])
            
            print('run', run, a.name)
        print('Run', run, ' finished')

        # End of i_episodes
    # End of runs

    d = logdir + 'asd'
    if os.path.exists(d):
        shutil.rmtree(d)

    logW = LogWriter(d, sync_cycle=100)

    for a_idx in range(len(names)):
        agent_runs = []

        for run in range(args.parralel_runs):
            agent_runs.append(all_ers[run][a_idx])

        ers_avg = np.sum(agent_runs, axis=0) / args.parralel_runs

        #log_seperate_agent(ers_avg, agent_name=names[a_idx])


        with logW.mode(names[a_idx]):
            l = logW.scalar('ers')

        for t, x in enumerate(ers_avg):
            l.add_record(t, x)


# run just one agent
# ==== MODE 1 ======
def evaluate():   
    all_ers = np.zeros(shape=(args.parralel_runs, args.n_episodes))
    
    for run in range(args.parralel_runs):
        start_run = time.time()
        agent = Agent(args, name='curious')

        for i_episode in range(args.n_episodes):
            start = time.time()
            agent.reset_env()
            is_done = False
            
            while not is_done:
                is_done = agent.play_step()
                
            ers = agent.ers[-1] 
            dqn_loss = agent.loss_dqn[-1]
            
            all_ers[run][i_episode] = agent.ers[-1]

            t = (time.time() - start)*1000

            if agent.args.has_curiosity:
                loss_combined = agent.loss_combined[-1]
                loss_inverse = agent.loss_inverse[-1] 
                cos_distance = agent.cos_distance[-1] 

                if args.debug:
                    print("p: {}   |     n: {}  |    epsilon: {:.2f}    |    dqn:  {:.2f}    |    ers:  {:.2f}    |    com: {:.2f}    |    inv: {:.2f}   cos: {:.2f}    |   time: {:.2f}".format(
                        multiprocessing.current_process().name, i_episode, agent.epsilon, dqn_loss, ers, loss_combined, loss_inverse, cos_distance, t))
            else:
                if args.debug:
                    print("{}   |    n: {}  |    epsilon: {:.2f}    |    dqn_loss:  {:.2f}    |    ers:  {}    |     time: {:.2f}".format(
                        args.device, i_episode, agent.epsilon, dqn_loss, ers, t))


        print('Run Nr: {}   |    Process id:{}   |    finished in {:.2f} s'.format(run, multiprocessing.current_process().name, (time.time() - start_run)))
        # End of i_episodes
    # End of runs


def init_child(lock_):
    global lock
    lock = lock_

# Runs multiple agents in parralel, then takes averages
def parralel_evaluate(params):
    grid_args = args
    grid_args.curiosity_beta = params['curiosity_beta']
    grid_args.curiosity_lambda = params['curiosity_lambda']
    grid_args.batch_size = params['batch_size']

    all_ers = np.zeros(shape=(args.parralel_runs, args.n_episodes))
    start_all = time.time()

    for run in range(args.parralel_runs):
        start_run = time.time()

        agent = Agent(grid_args, name='curious')

        for i_episode in range(args.n_episodes):
            agent.reset_env()
            is_done = False
            
            while not is_done:
                is_done = agent.play_step()

            if args.debug and i_episode % 10 == 0:
                print('i_episode:', i_episode, multiprocessing.current_process().name)

            all_ers[run][i_episode] = agent.ers[-1]

        if args.debug:     
            print('Run Nr: {}   |    Process id:{}   |    finished in {:.2f} s'.format(run, multiprocessing.current_process().name, (time.time() - start_run)))
        
        # End of i_episodes
    # End of runs

    #log_parralel_agents(ers_avg, ers_min, ers_max prefix='eval')

    # CSV logging
    data = params
    avg = all_ers.mean()
    data['ers_avg'] = [avg]
    df = pd.DataFrame(data)   
    fn = agent.args.env_name
    writefile(df, fn, lock)

    print('Logged: {} in {:.2f} s'.format(df.values[0], time.time() - start_all))

    

# ==== MODE 2 ======   
def multiprocess():
    curiosity_beta = np.round(np.arange(0, 1.1, 0.2), 1)
    curiosity_lambda = np.round(np.arange(0, 1.1, 0.2), 1)
    batch_size = [32]#, 64, 128, 256]
    param_grid = {'curiosity_beta': curiosity_beta, 'curiosity_lambda': curiosity_lambda, 'batch_size': batch_size}
    p = ParameterGrid(param_grid)
    p = list(p)

    lock = Lock()
    pool = Pool(processes=args.n_processes, initializer=init_child, initargs=(lock,))                                                        
    pool.map(parralel_evaluate, p) 
    #parralel_evaluate(p[0])

def writefile(df, file_name, lock):
    with lock:
        ex = os.getcwd() + "/" + file_name + ".csv"
        f = not os.path.exists(ex)
        df.to_csv(file_name + ".csv", mode='a', sep=',', header=f)
        

if __name__ == '__main__':
    main()
