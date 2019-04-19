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
parser.add_argument('-mode', type=int, default=1, help='0 - compare agents, 1 - run single , 2 - multiprocess grid search')
parser.add_argument('-device', default='cpu', help='cpu or cuda')

parser.add_argument('-debug', type=int, default=0, help='Extra print statements between episodes')
parser.add_argument('-debug_features', type=int, default=0, help='Use opencv to peer into feature states')
parser.add_argument('-debug_images', type=int, default=0, help='Use opencv to debug stacked frames')
parser.add_argument('-save_interval', type=int, default=100, help='Save model after n steps')
parser.add_argument('-save', type=int, default=0, help='Save models?')

parser.add_argument('-env_name', required=True,  help='OpenAI game enviroment name')
parser.add_argument('-learning_rate', type=float, default=0.001)
parser.add_argument('-batch_size', type=int, default=32)

parser.add_argument('-has_normalized_state', type=int, default=0, help='Normalize state vector in forward and inverse models? true | false')
parser.add_argument('-epsilon_decay', type=float, required=True, help='Epslion decay for exploration / explotation policy')
parser.add_argument('-epsilon_floor', type=float, default=0.01, help='Where epsilon stops to decay')
parser.add_argument('-gamma', type=float, default=0.95, help='Hyperparameter for DQN')
parser.add_argument('-n_episodes', type=int, default=50, help='Number of episodes (games) to be played')
parser.add_argument('-n_frames', type=int, default=9999, help='Number of frames per one episode')
parser.add_argument('-memory_size', type=int, default=10000, help="Replay memory size (This code uses sum tree, not deque)")

parser.add_argument('-has_images', type=int, default=0, help='Whether or not the game state is an image')
parser.add_argument('-image_scale', type=float, default=1.0, help='Image downscaling factor')
parser.add_argument('-n_sequence_stack', type=int, default=4, help='How many frames are in frame stack (deque)')
parser.add_argument('-n_frame_skip', type=int, default=4, help='How many frames to skip, before pushing to frame stack')
parser.add_argument('-image_crop', type=int, nargs='+', help='Coordinates to crop image, x1 y1 x2 y2')

parser.add_argument('-parralel_runs', type=int, default=5, help="How many parralel agents to simulate")
parser.add_argument('-n_processes', type=int, default=3, help="How many parralel processes to run (MODE 3)")

parser.add_argument('-state_min_val', type=float, default=-1.0, help="Manual min value for feature encoder normalization")
parser.add_argument('-state_max_val', type=float, default=-1.0, help="Manual max value for feature encoder normalization")

parser.add_argument('-has_curiosity', type=int, required=True)
parser.add_argument('-curiosity_beta', type=float, default=-1.0, help='Beta hyperparameter for curiosity module')
parser.add_argument('-curiosity_lambda', type=float, default=-1.0, help='Lambda hyperparameter for curiosity module')
parser.add_argument('-curiosity_scale', type=float, default=1.0, help='Intrinsic reward scale factor')

parser.add_argument('-encoder_1_layer_out', type=int, default=5)
parser.add_argument('-encoder_2_layer_out', type=int, default=10)
parser.add_argument('-encoder_3_layer_out', type=int, default=15)

parser.add_argument('-inverse_1_layer_out', type=int, default=30)
parser.add_argument('-inverse_2_layer_out', type=int, default=20)

parser.add_argument('-forward_1_layer_out', type=int, default=30)
parser.add_argument('-forward_2_layer_out', type=int, default=20)

parser.add_argument('-dqn_1_layer_out', type=int, default=64)
parser.add_argument('-dqn_2_layer_out', type=int, default=32)

parser.add_argument('-has_ddqn', type=int, default=0, help='Is double DQN enabled?')
parser.add_argument('-target_update', type=float, default=10, help='Update target network after n steps')

args = parser.parse_args()

logdir = "./logs/"
save_dir = "save"

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

def get_cleaned_logger(folder_name):
    d = logdir + folder_name
    if os.path.exists(d):
        shutil.rmtree(d)

    return LogWriter(d, sync_cycle=1000)

def log_evaluate_agent(ers_avg, folder_name):
    logW = get_cleaned_logger(folder_name)


def log_comparison_agents(all_ers, names, folder_name):
    logW = get_cleaned_logger(folder_name)
    data = {}

    for a_idx in range(len(names)):
        agent_runs = []

        for run in range(args.parralel_runs):
            agent_runs.append(all_ers[run][a_idx])

        ers_avg = np.sum(agent_runs, axis=0) / args.parralel_runs
        ers_min = np.min(agent_runs, axis=0)
        ers_max = np.max(agent_runs, axis=0)

        with logW.mode(names[a_idx]):
            l = logW.scalar('ers_' + args.env_name)

        for t, x in enumerate(ers_avg):
            l.add_record(t, x)
        
        data[names[a_idx] + "_ers"] = ers_avg
        data[names[a_idx] + "_min"] = ers_min
        data[names[a_idx] + "_max"] = ers_max

    file_name = "ers_data.csv"
    ex = logdir + folder_name + "/" + file_name
    f = not os.path.exists(ex)

    df = pd.DataFrame(data)
    df.to_csv(ex, mode='a', sep=',', header=f)

def save_model(agent, run, i_episode, folder_name):
    if i_episode % args.save_interval == 0:
        folder = os.path.join(save_dir, folder_name)
        
        models = {}
        models['dqn'] = agent.dqn_model

        if agent.args.has_curiosity:
            models['inverse'] = agent.inverse_model
            models['forward'] = agent.forward_model
            models['encoder'] = agent.encoder_model

        for name in models:
            fn = os.path.join(folder, name)
            if not os.path.exists(fn):
                os.makedirs(fn)

        for name, model in models.items():
            file_name = "{}__run_{}__i_ep_{}".format(name, run, i_episode)
            path = os.path.join(os.getcwd(), folder, name, file_name)
            torch.save(model.state_dict(), path)
        

        print("saved checkpoint ar run: {}  |  i_episode: {}".format(run, i_episode))

# This mode compares n agents
# ==== MODE 0 ======
def comparison():
    # -------------------- Write all agents to test in here ----------------
    names = ['curious', 'curious_ddqn', 'dqn', 'ddqn']

    args.has_curiosity = 0

    curious_args = args
    curious_args.has_curiosity = 1

    curious_ddqn_args = args
    curious_ddqn_args.has_curiosity = 1
    curious_ddqn_args.has_ddqn = 1

    ddqn_args = args
    ddqn_args.has_ddqn = 1

    all_args = [curious_args, curious_ddqn_args, args, ddqn_args]

    # -----------------------------------------------------------------------
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
                if i_episode % 10 == 0:
                    print(i_episode, a.ers[-1])
            
            print('run', run, a.name)
        print('Run', run, ' finished')

        # End of i_episodes
    # End of runs

    log_comparison_agents(all_ers, names, folder_name='comp')



# run just one agent
# ==== MODE 1 ======
def evaluate():   
    save_folder = datetime.datetime.now().strftime("%b-%d-%H:%M")
    all_ers = np.zeros(shape=(args.parralel_runs, args.n_episodes))

    logW = get_cleaned_logger(folder_name='eval')
   
    for run in range(args.parralel_runs):
        start_run = time.time()
        agent = Agent(args, name='curious')
        
        with logW.mode('run: ' + str(run)):
            l = logW.scalar('ers')

        for i_episode in range(args.n_episodes):
            start = time.time()
            agent.reset_env()
            is_done = False
            
            while not is_done:
                #agent.env.render()
                is_done = agent.play_step()
                
                             
            all_ers[run][i_episode] = agent.ers[-1]
            t = time.time() - start
            agent.print_debug(i_episode, exec_time=t)

            l.add_record(i_episode, agent.ers[-1])

            save_model(agent, run, i_episode, folder_name=save_folder)

        print('Run Nr: {}   |    Process id:{}   |    finished in {:.2f} s'.format(run, multiprocessing.current_process().name, (time.time() - start_run)))

    with logW.mode('avg_runs: '):
        l = logW.scalar('ers')

    all_ers = np.array(all_ers)
    ers_avg = np.sum(all_ers, axis=0) / args.parralel_runs

    for t, x in enumerate(ers_avg):
        l.add_record(t, x)

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

    # CSV logging
    data = params
    avg = all_ers.mean()
    data['ers_avg'] = [avg]
    df = pd.DataFrame(data)   
    fn = agent.args.env_name
    writefile(df, fn, lock)

    print('{} logged: {} in {:.2f} s'.format(args.env_name, df.values[0], time.time() - start_all))

    

# ==== MODE 2 ======   
def multiprocess():
    curiosity_beta = np.round(np.arange(0, 1.1, 0.1), 1)
    curiosity_lambda = np.round(np.arange(0, 1.1, 0.1), 1)
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
