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
parser.add_argument('-m', '--mode', type=int, default=1, help='0 - compare two models, 1 - run single model, 2 - DQN vs DDQN comparison, 3 - multiprocess testing')
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
parser.add_argument('--parralel_runs', type=int, default=10, help="How many parralel agents to simulate")
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

# Log n parralel runs from timeline
def log_seperate_timelines(timelines, agent_names, prefix, process_id=0):
    d = logdir + prefix
    logW = LogWriter(d, sync_cycle=1000)

    for timeline, agent_name in zip(timelines, agent_names):
        with logW.mode(agent_name):
            logger_ers_min = logW.scalar("/ers/min")
            logger_ers_avg = logW.scalar("/ers/avg")
            logger_ers_max = logW.scalar("/ers/max")

        for parralel_time_slices in timeline:
            parralel_min = min([time_slice.ers_avg for time_slice in parralel_time_slices])
            parralel_max = max([time_slice.ers_avg for time_slice in parralel_time_slices])
            parralel_avg = sum([time_slice.ers_avg for time_slice in parralel_time_slices]) / len(parralel_time_slices)

            logger_ers_min.add_record(parralel_time_slices[0].i_episode, parralel_min)
            logger_ers_avg.add_record(parralel_time_slices[0].i_episode, parralel_max)
            logger_ers_max.add_record(parralel_time_slices[0].i_episode, parralel_avg)



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

    # CONCACENTAQTE COLUMN WISE AGENT ERS ETC

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
        
        
   


# This mode compares two agents
# args.parralel_runs is how many parralel agents are simulated and then taken average of
# ==== MODE 0 ======
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

        for i_episode in range(1, args.n_episodes + 1):
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
    
    log_seperate_timelines(timelines, agent_names=timeline_names, prefix='comp')




# run just one agent
# ==== MODE 1 ======
def evaluate():   
    all_ers = np.zeros(shape=(args.parralel_runs, args.n_episodes))
    
    for run in range(args.parralel_runs):
        start_run = time.time()
        agent = Agent(args, name='curious')

        for i_episode in range(1, args.n_episodes + 1):
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
                        idx, i_episode, agent.epsilon, dqn_loss, ers, loss_combined, loss_inverse, cos_distance, t))
            else:
                if args.debug:
                    print("{}   |    n: {}  |    epsilon: {:.2f}    |    dqn_loss:  {:.2f}    |    ers:  {}    |     time: {:.2f}".format(
                        args.device, i_episode, agent.epsilon, dqn_loss, ers, t))


        print('Run Nr: {}   |    Process id:{}   |    finished in {:.2f} s'.format(run, multiprocessing.current_process().name, (time.time() - start_run)))
        # End of i_episodes
    # End of runs

  
 # ==== MODE 2 ======
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

    for run in range(args.parralel_runs):
        start_run = time.time()

        agent = Agent(grid_args, name='curious')

        for i_episode in range(0, args.n_episodes):
            agent.reset_env()
            is_done = False
            
            while not is_done:
                is_done = agent.play_step()

            all_ers[run][i_episode] = agent.ers[-1]
                
        print('Run Nr: {}   |    Process id:{}   |    finished in {:.2f} s'.format(run, multiprocessing.current_process().name, (time.time() - start_run)))
        
        # End of i_episodes
    # End of runs

    #log_parralel_agents(ers_avg, ers_min, ers_max prefix='eval')

    # CSV logging
    data = params
    avg = all_ers.mean()
    data['ers_avg'] = [avg]
    df = pd.DataFrame(data)   
    fn = args.env_name
    writefile(df, fn, lock)



# ==== MODE 3 ======   
def multiprocess():
    curiosity_beta = np.round(np.arange(0, 1.1, 0.1), 1)
    curiosity_lambda = np.round(np.arange(0, 1.1, 0.1), 1)
    batch_size = [32]#, 64, 128, 256]
    env_name = args.env_name
    param_grid = {'curiosity_beta': curiosity_beta, 'curiosity_lambda': curiosity_lambda, 'batch_size': batch_size, 'env_name': env_name}
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
