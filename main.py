# pip install box2d-py, gym, pandas, visualdl, opencv-python

import logging, Box2D, os, shutil, datetime, time, random, sys, torch, argparse, gym, pandas, copy
from gym import envs
import numpy as np

from agent import Agent
from pyvirtualdisplay import Display


from modules.args_utils import ArgsUtils
from modules.csv_utils import CsvUtils
from modules.file_utils import FileUtils
from modules.logging_utils import LoggingUtils

def arg_to_bool(x): return str(x).lower() == 'true'

parser = argparse.ArgumentParser()
parser.add_argument('-device', default='cpu', help='cpu or cuda')

parser.add_argument('-debug', default=False, type=arg_to_bool, help='Extra print statements between episodes')
parser.add_argument('-debug_features', default=False, type=arg_to_bool, help='Use opencv to peer into feature states')
parser.add_argument('-debug_images', default=False, type=arg_to_bool, help='Use opencv to debug stacked frames')
parser.add_argument('-debug_activations', default='0 0 0 0', type=str, nargs='+', help='Show activation maps. Args: denseblock denselayer convlayer. Example: 1 13 2')

parser.add_argument('-save_interval', default=100, type=int, help='Save model after n steps')
parser.add_argument('-load_path', default='', help='folder name from where to load, last episode will be taken')

parser.add_argument('-env_name', default='CartPole-v0', help='OpenAI game enviroment name')
parser.add_argument('-learning_rate', default=1e-3, type=float)
parser.add_argument('-decoder_coeficient', default=1e-2, type=float, help='How much is decoder used in training 0..1')
parser.add_argument('-batch_size', default=32, type=int)

parser.add_argument('-is_normalized_state', default=False, type=arg_to_bool, help='Normalize state vector in forward and inverse models? true | false')
parser.add_argument('-epsilon_decay', default=1e-4, type=float, help='Epslion decay for exploration / explotation policy')
parser.add_argument('-epsilon_floor', default=0.01, type=float, help='Where epsilon stops to decay')
parser.add_argument('-epsilon_start', default=1.0, type=float)
parser.add_argument('-gamma', default=0.95, type=float, help='Hyperparameter for DQN')
parser.add_argument('-n_episodes', default=500, type=int, help='Number of episodes (games) to be played')
parser.add_argument('-n_frames', default=9999, type=int, help='Number of frames per one episode')

parser.add_argument('-offline_iterations', default=1, type=int)

parser.add_argument('-memory_size', default=100, type=int, help="Replay memory size (This code uses sum tree, not deque)")
parser.add_argument('-prioritized_type', default='rank', help='random / proportional / rank')
parser.add_argument('-rank_update', default=10, type=int, help='After how many steps is memory sorted(only for rank prioritization)')
parser.add_argument('-per_e', default=0.01, type=float, help='Hyperparameter that we use to avoid some experiences to have 0 probability of being taken')
parser.add_argument('-per_a', default=0.9, type=float, help='Hyperparameter that we use to make a tradeoff between taking only exp with high priority and sampling randomly')
parser.add_argument('-per_b', default=0.1, type=float, help='Importance-sampling, from initial value increasing to 1')
parser.add_argument('-per_b_anneal_to', default=100000, type=int, help='At which frame does beta anneal to 1.0')

parser.add_argument('-image_scale', default=1.0, type=float, help='Image downscaling factor')
parser.add_argument('-n_frame_skip', default=1, type=int, help='How many frames to skip, before pushing to frame stack')
parser.add_argument('-image_crop', default=0, type=int, nargs='+', help='Coordinates to crop image, x1 y1 x2 y2')
parser.add_argument('-is_grayscale', default=False, type=arg_to_bool, help='Whether state image is converted from RGB to grayscale ')

parser.add_argument('-is_curiosity', default=True, type=arg_to_bool)
parser.add_argument('-curiosity_beta', default=0.5, type=float, help='Beta hyperparameter for curiosity module')
parser.add_argument('-curiosity_lambda', default=0.5, type=float, help='Lambda hyperparameter for curiosity module')
parser.add_argument('-curiosity_scale', default=100.0, type=float, help='Intrinsic reward scale factor')

# if simple or conv autoencoder will be included
# important encoder_warmup_dqn_reset_steps and encoder_warmup_dqn_reset_steps_end
parser.add_argument('-encoder_type', default='simple', nargs='?', choices=['nothing', 'simple', 'conv'], help='Which type od encoder to use, depends on game state (default: %(default)s)')
parser.add_argument('-n_sequence', default=1, type=int, help='How many stacked states will be passed to encoder')
parser.add_argument('-encoder_warmup_dqn_reset_steps', default=500, type=int) # warmup autoencoder
parser.add_argument('-encoder_warmup_dqn_reset_steps_end', default=3000, type=int)  # warmup autoencoder
parser.add_argument('-encoder_warmup_lock', default=False, type=arg_to_bool)

parser.add_argument('-encoding_size', type=int, default=4)
parser.add_argument('-models_layer_count', type=int, default=2, help='Hidden layer count for inverse / forward / dqn / simple encoder models')
parser.add_argument('-models_layer_features', type=int, default=16, help='Hidden layer FEATURE count for inverse / forward / dqn / simple encoder models')
parser.add_argument('-simple_encoder_layers', type=int, default=[16, 8], nargs="+", help='How many outputs per each layer e.g. 256 64 32')
parser.add_argument('-rnn_layers', type=int, default=1, help='How many hidden layers in LSTM')

parser.add_argument('-conv_encoder_layer_out', default=1024, type=int)
parser.add_argument('-render_xvfb', default=False, type=arg_to_bool, help='wether to render games like cart pole as an image')

parser.add_argument('-is_ddqn', type=arg_to_bool, default=True, help='Is double DQN enabled?')
parser.add_argument('-target_update', default=100, type=int, help='Update target network after n steps')

parser.add_argument('-id', default=0, type=int)
parser.add_argument('-repeat_id', default=0, type=int)
parser.add_argument('-report', default='report', type=str)
parser.add_argument('-params_report', nargs='*', required=False)
parser.add_argument('-name', help='Run name, by default date', default='test', type=str)

args, args_other = parser.parse_known_args()

tmp = [
    'episode',
    'e_score', # add extra params that you are interested in
    'e_score_min',
    'e_score_max',
    'score_avg',
    'score_best',
    'loss',
    'loss_dqn',
    'loss_enc',
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
logging.info('global args ok')

CsvUtils.create_local(args)

if args.render_xvfb:
    display = Display(visible=0, size=(1400, 900))
    display.start()


    

if torch.cuda.is_available():
    logging.info('cuda detected')

def main():
    evaluate()


def save(agent, i_episode, dir_name):
    path = f'./save/{dir_name}/{i_episode}/'
    FileUtils.createDir(path)

    if not os.path.isfile(path + 'args.txt'):
        with open(path + "args.txt", "w") as f:
            for arg in vars(args):
                key = arg
                value = getattr(args, arg)
                if isinstance(value, list):
                    value = ' '.join([str(it) for it in value])
                f.write(f"{arg} : {value}\n")

    torch.save(agent.optimizer_agent.state_dict(), path + 'optimizer_agent.pth')
    
    torch.save(agent.dqn_model.state_dict(), path + 'dqn.pth')
    torch.save(agent.target_model.state_dict(), path + 'dqn_target.pth')

    if agent.args.is_curiosity:
        torch.save(agent.forward_model.state_dict(), path + 'forward.pth')
        torch.save(agent.inverse_model.state_dict(), path + 'inverse.pth')
    
    if agent.args.encoder_type != 'nothing':
        torch.save(agent.feature_encoder.encoder.state_dict(), path + 'encoder.pth')
        torch.save(agent.feature_decoder.decoder.state_dict(), path + 'decoder.pth')
        torch.save(agent.optimizer_autoencoder.state_dict(), path + 'optimizer_autoencoder.pth')

def evaluate():   
    now = datetime.datetime.now()
    dir_name = now.strftime("%B_%d_at_%H_%M_%p")

    agent = Agent(args, name='curious')
    logging.info('Agent created, starting training')
        
    for i_episode in range(args.n_episodes):
        start = time.time()
        agent.reset_env()
        is_done = False
        
        while not is_done:
            is_done = agent.play_step()
                        
        t = time.time() - start
        
        state = agent.get_results()
        CsvUtils.add_results_local(args, state)

        if args.debug:
            logging.info(agent.print_debug(i_episode, t))

        # if (i_episode + 1) % args.save_interval == 0:
        save(agent, i_episode + 1, dir_name)

    state = agent.get_results()
    CsvUtils.add_results(args, state)
    logging.info(f'Report: {args.id}  |   finished in {t:.2f} s')
    agent.env.close()


if __name__ == '__main__':
    main()

