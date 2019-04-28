# pip install box2d-py, gym, pandas, visualdl, opencv-python
import logging

from visualdl import LogWriter # Has to be imported before everython elese, otherwise dumps
import Box2D, os, shutil, datetime, time, random, sys, torch, argparse, gym, pandas, copy
from gym import envs
import numpy as np
import multiprocessing
from sklearn.model_selection import ParameterGrid
from multiprocessing import Pool, Process, Lock
import pandas as pd

from agent_dqn import AgentDQN

from modules.args_utils import ArgsUtils
from modules.csv_utils import CsvUtils
from modules.file_utils import FileUtils
from modules.logging_utils import LoggingUtils

def arg_to_bool(x): return str(x).lower() == 'true'

parser = argparse.ArgumentParser()
parser.add_argument('-mode', default=1, type=int, help='0 - compare agents, 1 - run single , 2 - multiprocess grid search')
parser.add_argument('-device', default='cpu', help='cpu or cuda')

parser.add_argument('-debug', default=False, type=arg_to_bool, help='Extra print statements between episodes')
parser.add_argument('-debug_features', default=False, type=arg_to_bool, help='Use opencv to peer into feature states')
parser.add_argument('-debug_images', default=False, type=arg_to_bool, help='Use opencv to debug stacked frames')
parser.add_argument('-debug_activations', type=str, nargs='+', help='Show activation maps. Args: denseblock denselayer convlayer. Example: 1 13 2')

parser.add_argument('-save_interval', default=100, type=int, help='Save model after n steps')

parser.add_argument('-env_name', required=True, help='OpenAI game enviroment name')
parser.add_argument('-learning_rate', default=0.001, type=float)
parser.add_argument('-batch_size', default=32, type=int)

parser.add_argument('-is_normalized_state', default=False, type=arg_to_bool, help='Normalize state vector in forward and inverse models? true | false')
parser.add_argument('-epsilon_decay', required=True, type=float, help='Epslion decay for exploration / explotation policy')
parser.add_argument('-epsilon_floor', default=0.01, type=float, help='Where epsilon stops to decay')
parser.add_argument('-gamma', default=0.95, type=float, help='Hyperparameter for DQN')
parser.add_argument('-n_episodes', default=50, type=int, help='Number of episodes (games) to be played')
parser.add_argument('-n_frames', default=9999, type=int, help='Number of frames per one episode')

parser.add_argument('-is_prioritized', default=True, type=arg_to_bool, help='Is priositized experience replay beeing used?')
parser.add_argument('-memory_size', default=10000, type=int, help="Replay memory size (This code uses sum tree, not deque)")

parser.add_argument('-image_scale', default=1.0, type=float, help='Image downscaling factor')
parser.add_argument('-n_sequence', default=4, type=int, help='How many stacked states will be passed to encoder')
parser.add_argument('-n_frame_skip', default=1, type=int, help='How many frames to skip, before pushing to frame stack')
parser.add_argument('-image_crop', type=int, nargs='+', help='Coordinates to crop image, x1 y1 x2 y2')
parser.add_argument('-is_grayscale', default=False, type=arg_to_bool, help='Whether state image is converted from RGB to grayscale ')

parser.add_argument('-is_curiosity', default=False, type=arg_to_bool, required=True)
parser.add_argument('-curiosity_beta', default=-1.0, type=float, help='Beta hyperparameter for curiosity module')
parser.add_argument('-curiosity_lambda', default=-1.0, type=float, help='Lambda hyperparameter for curiosity module')
parser.add_argument('-curiosity_scale', default=1.0, type=float, help='Intrinsic reward scale factor')

parser.add_argument('-encoder_type', default='nothing', nargs='?', choices=['nothing', 'simple', 'conv'], help='Which type od encoder to use, depends on game state (default: %(default)s)')
parser.add_argument('-simple_encoder_1_layer_out', default=5, type=int)
parser.add_argument('-simple_encoder_2_layer_out', default=5, type=int)

parser.add_argument('-conv_encoder_layer_out', default=15, type=int,)

parser.add_argument('-inverse_1_layer_out', default=30, type=int,)
parser.add_argument('-inverse_2_layer_out', default=25, type=int,)

parser.add_argument('-forward_1_layer_out', default=30, type=int)
parser.add_argument('-forward_2_layer_out', default=20, type=int)

parser.add_argument('-dqn_1_layer_out', default=15, type=int,)
parser.add_argument('-dqn_2_layer_out', default=15, type=int,)
parser.add_argument('-dqn_3_layer_out', default=10, type=int,)
parser.add_argument('-dqn_4_layer_out', default=10, type=int,)

parser.add_argument('-is_ddqn', type=arg_to_bool, default=False, help='Is double DQN enabled?')
parser.add_argument('-target_update', default=10, type=int, help='Update target network after n steps')

parser.add_argument('-id', default=0, type=int)
parser.add_argument('-repeat_id', default=0, type=int)
parser.add_argument('-report', default='report', type=str)
parser.add_argument('-params_report', nargs='*', required=False)
parser.add_argument('-name', help='Run name, by default date', default='test', type=str)

args, args_other = parser.parse_known_args()

logdir = "./logs/"
save_dir = "save"

tmp = [
    'episode',
    'e_score', # add extra params that you are interested in
    'e_score_min',
    'e_score_max',
    'score_avg',
    'score_best',
    'loss',
    'loss_dqn',
    'loss_inverse',
    'loss_forward', 
    'cosine_distance'
]
if not args.params_report is None:
    for it in reversed(args.params_report):
        if not it in tmp:
            tmp.insert(0, it)

args.params_report = tmp
args.params_report_local = args.params_report


FileUtils.createDir('./tasks/' + args.report)
run_path = './tasks/' + args.report + '/runs/' + args.name

if os.path.exists(run_path):
    shutil.rmtree(run_path, ignore_errors=True)
    time.sleep(3)
    while os.path.exists(run_path):
        pass

FileUtils.createDir(run_path)
logging_utils = LoggingUtils(filename=os.path.join(run_path, 'log.txt'))
is_logged_cnorm = False

ArgsUtils.log_args(args, 'main.py', logging_utils)

CsvUtils.create_local(args)

if torch.cuda.is_available():
    print('cuda detected')

mode = { 
        0: 'comparison',
        1: 'evaluate',
    }

def main():
    print('Running mode: {}'.format(mode[args.mode]))
    run = {
        0: comparison,
        1: evaluate,
        }
    
    run[args.mode]()

def get_cleaned_logger(folder_name):
    d = logdir + folder_name
    if os.path.exists(d):
        shutil.rmtree(d)

    return LogWriter(d, sync_cycle=1000)

def log_evaluate_agent(ers_avg, folder_name):
    logW = get_cleaned_logger(folder_name)


def log_comparison_agents(all_ers, folder_name):
    logW = get_cleaned_logger(folder_name)
    data = {}

    for agent_name, ers in all_ers.items():
        ers = np.array(ers)
    
        score_avg = np.sum(ers, axis=0) / args.parralel_runs
        score_std = np.std(ers, axis=0)

        with logW.mode(agent_name):
            l = logW.scalar('avg_score_' + args.env_name)

        for t, x in enumerate(score_avg):
            l.add_record(t, x)
        
        data[agent_name + "_avg_score"] = score_avg
        data[agent_name + "_std"] = score_std

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

        if agent.args.is_curiosity:
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
        
        #TODO use logging.info(f'..')
        logging.info(f'saved checkpoint ar run: {run}  |  i_episode: {i_episode}')

# This mode compares n agents
# ==== MODE 0 ======
def comparison():
    # Args are passed as references, so deep copy is requred
    agents = {}

    args.is_curiosity = False
    args.is_ddqn = False
    
    ddqn_args = copy.deepcopy(args)
    ddqn_args.is_ddqn = True

    curious_args = copy.deepcopy(args)
    curious_args.is_curiosity = True
    
    curious_ddqn_args = copy.deepcopy(args)
    curious_ddqn_args.is_curiosity = True
    curious_ddqn_args.is_ddqn = True

    all_ers = {}

    for n in ['dqn', 'ddqn', 'curious', 'curious_ddqn']:
        all_ers[n] = np.zeros(shape=(args.parralel_runs, args.n_episodes)) 
   
    for run in range(args.parralel_runs):
        agents['dqn'] = AgentDQN(args, name='dqn')
        agents['ddqn'] = AgentDQN(ddqn_args, name='ddqn')
        agents['curious'] = AgentCurious(curious_args, name='curious')
        agents['curious_ddqn'] = AgentCurious(curious_ddqn_args, name='curious_ddqn')

        for n, a in agents.items():
            for i_episode in range(args.n_episodes):
                a.reset_env()
                is_done = False

                while not is_done:
                    is_done = a.play_step()
                
                all_ers[n][run][i_episode] = np.array(sum(a.e_reward))
            
            print('run', run, a.name)
        print('Run', run, ' finished')

        # End of i_episodes
    # End of runs

    log_comparison_agents(all_ers, folder_name='comp')



# run just one agent
# ==== MODE 1 ======
def evaluate():   
    save_folder = datetime.datetime.now().strftime("%b-%d-%H:%M")
    all_ers = np.zeros(shape=(args.n_episodes))
   
    agent = AgentDQN(args, name='curious')
        
    for i_episode in range(args.n_episodes):
        start = time.time()
        agent.reset_env()
        is_done = False
        
        while not is_done:
            is_done = agent.play_step()
                         
        all_ers[i_episode] = sum(agent.e_reward)
        t = time.time() - start

        state = agent.get_results()
        CsvUtils.add_results_local(args, state)

        if args.debug:
            logging.info(agent.print_debug(i_episode, t))

    state = agent.get_results()
    CsvUtils.add_results(args, state)
    logging.info(f'Report: {args.id}  |   finished in {t:.2f} s')


if __name__ == '__main__':
    main()
