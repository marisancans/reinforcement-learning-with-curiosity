import typing
from sklearn.model_selection import ParameterGrid
import subprocess
import traceback
import logging
import os
import re
import sys
import time
import copy
import json
import platform
import numpy as np
import argparse
from datetime import datetime

from modules.file_utils import FileUtils
from modules.logging_utils import LoggingUtils
from modules.args_utils import ArgsUtils
from sklearn.model_selection import ParameterGrid
import subprocess



parser = argparse.ArgumentParser(description='task generator (HPC and local)')

parser.add_argument(
    '-main_script',
    default='main.py',
    type=str)

parser.add_argument(
    '-conda_env',
    help='name of conda environment',
    default='conda_env',
    type=str)

parser.add_argument(
    '-repeat',
    help='how many times each set of parameters should be repeated for testing stability',
    default=1,
    type=int)

parser.add_argument(
    '-report',
    help='csv report of all tasks combined',
    default='tasks',
    type=str)

parser.add_argument(
    '-params_report',
    help='csv columns, parameters for summary',
    default=['id', 'name', 'repeat_id'],
    nargs='*',
    required=False) # extra params for report header

parser.add_argument(
    '-params_grid',
    nargs='*',
    help='parameters for grid search',
    required=False) # for search

parser.add_argument(
    '-process_count_per_task',
    help='how many simulations/parallel tasks should be run per task',
    default=1,
    type=int)

parser.add_argument(
    '-is_hpc',
    help='is HPC qsub tasks or local tasks',
    default=True,
    type=lambda x: (str(x).lower() == 'true'))

parser.add_argument(
    '-hpc_queue',
    help='hpc queue',
    default='batch',
    type=str)

parser.add_argument(
    '-hpc_feautre',
    help='k40 v100',
    default='v100',
    type=str)

parser.add_argument(
    '-is_xvfb',
    help='is headless x-window display created for task',
    default=False,
    type=lambda x: (str(x).lower() == 'true'))

parser.add_argument(
    '-hpc_gpu_count',
    help='HPC - how many GPUs used per task',
    default=1,
    type=int)

parser.add_argument(
    '-hpc_cpu_count',
    help='HPC - how many CPUs used per task',
    default=8,
    type=int)

parser.add_argument(
    '-single_task',
    help='for testing generate only single task (debug)',
    default=False,
    type=lambda x: (str(x).lower() == 'true'))

args, args_other = parser.parse_known_args()
args = ArgsUtils.add_other_args(args, args_other)
args_other_names = ArgsUtils.extract_other_args_names(args_other)


# add all testable parameters to final report header
args.params_report += args_other_names

FileUtils.createDir('./reports')
FileUtils.createDir('./tasks')
FileUtils.createDir(f'./tasks/{args.report}')
if args.is_hpc:
    FileUtils.createDir(os.path.expanduser('~') + '/tmp')

logging_utils = LoggingUtils(filename=os.path.join('reports', args.report + '.txt'))
ArgsUtils.log_args(args, 'taskgen.py', logging_utils)

# read current progress of task IDs so that you can track every ID of tasks
task_settings = {
    'id': 0,
    'repeat_id': 0
}
path_task_settings_json = f'./tasks/tasks.json'
if os.path.exists(path_task_settings_json):
    with open(path_task_settings_json, 'r') as outfile:
        hpc_settings_loaded = json.load(outfile)
        for key in hpc_settings_loaded:
            task_settings[key] = hpc_settings_loaded[key]

formated_params_grid = {}
formated_params_seq = {}

if not args.params_grid is None:
    for key_grid in args.params_grid:
        formated_params_grid[key_grid] = []

        for arg in vars(args):
            key = arg
            value = getattr(args, arg)

            if key == key_grid:
                if value is None:
                    raise Exception('Missing values for grid search key: {}'.format(key_grid))
                if len(value) < 2:
                    raise Exception('Not enough grid search values for key: {}'.format(key_grid))
                else:
                    formated_params_grid[key_grid] += value
                break


for arg in vars(args):
    key = arg
    value = getattr(args, arg)

    if key in args_other_names:
        if not key in formated_params_grid:
            if not value is None and len(value) > 0 and not value[0] is None:
                formated_params_seq[key] = value

            if len(value) > 1:
                logging_utils.info(f'Not in grid: {key}')
                logging_utils.info(json.dumps(formated_params_seq[key], indent=4))

grid = []
if len(list(formated_params_grid)) > 0:
    grid = list(ParameterGrid(formated_params_grid))


# add sequences
for each_seq in formated_params_seq:
    if len(formated_params_seq[each_seq]) > 1:
        for value in formated_params_seq[each_seq]:
            grid.append({
                each_seq: value
            })

if len(grid) == 0:
    grid.append({})

# add const params
for each_seq in formated_params_seq:
    value = formated_params_seq[each_seq]
    if len(value) == 1:
        for each_grid in grid:
            each_grid[each_seq] = value[0]

path_base = os.path.dirname(os.path.abspath(__file__))

tmp = ['id', 'name', 'repeat_id']
if not args.params_report is None:
    for it in args.params_report:
        if not it in tmp:
            tmp.append(it)
args.params_report = tmp

xvfb_disp_id = 1
hpc_gpu_queue = 0
any_count = 0
max_count = len(grid) * args.repeat
logging_utils.info('{} total tasks {}'.format(args.report, max_count))

script_path = ''
process_per_task = 0


logging_utils.info('tasks summary:')
tmp_id = task_settings['repeat_id']
tmp_cnt = 0
if args.single_task:
    grid = grid[:1]

for params_comb in grid:
    tmp_id += 1
    tmp_cnt += 1

logging_utils.info(f'\n\n{tmp_cnt} / {len(grid)}: {tmp_id}')
logging.info(f'formated_params_grid:{json.dumps(formated_params_grid, indent=4)}')

print('are tests ok? proceed?')
if input('[y/n]: ') != 'y':
    exit()

windows_log_list = []

for idx_comb, params_comb in enumerate(grid):
    task_settings['repeat_id'] += 1

    params_comb['report'] = args.report
    params_comb['id'] = args.report
    params_comb['params_report'] = ' '.join(args.params_report)
    params_comb['repeat_id'] = task_settings['repeat_id']

    for idx_repeat in range(args.repeat):
        task_settings['id'] += 1

        # save settings so that task IDs always are unique
        with open(path_task_settings_json, 'w') as outfile:
            json.dump(task_settings, outfile)

        params_comb['id'] = task_settings['id']
        params_comb['name'] = args.report + '_' + str(task_settings['repeat_id']) + '_' + str(task_settings['id'])

        # how many paralell processes to execute
        max_process_per_task = args.process_count_per_task

        is_hpc_gpu = False
        if args.is_hpc:
            params_comb['device'] = 'cpu'
            if args.hpc_feautre == 'v100' or args.hpc_feautre == 'k40':
                params_comb['device'] = 'cuda'
                is_hpc_gpu = True

        str_params = []
        for key in params_comb:
            value_param = params_comb[key]
            if isinstance(value_param, typing.List):
                value_param = ' '.join(value_param)
            str_params.append('-' + key + ' ' + str(value_param)) # format of parameters -param value
        str_params = ' '.join(str_params)

        is_windows = False
        script_ext = '.sh'
        if platform.system().lower() == 'windows':
            script_ext = '.bat'
            is_windows = True

        each_task_line_end = ''
        if max_process_per_task > 1:
            if not is_windows:
                each_task_line_end = '&'

        if process_per_task == 0:
            script_path = f'{path_base}/tasks/{args.report}/' + params_comb['name'] + script_ext

        with open(script_path, 'w' if process_per_task == 0 else 'a') as fp:
            if process_per_task == 0: # header before first script

                if args.is_hpc:
                    fp.write(f'#!/bin/sh -v\n')
                    fp.write(f'#PBS -e {path_base}/tasks/{args.report}\n') #stdout
                    fp.write(f'#PBS -o {path_base}/tasks/{args.report}\n') #errout

                    fp.write(f'#PBS -q {args.hpc_queue}\n')

                    cpu_count = args.hpc_cpu_count #* max_process_per_task
                    feature = ''
                    if len(args.hpc_feautre) > 0:
                        feature = f',feature={args.hpc_feautre}'

                    fp.write(f'#PBS -p 1000\n')
                    if is_hpc_gpu:
                        shared_setting = ''
                        if max_process_per_task > 1:
                            shared_setting = ':shared'
                        fp.write(f'#PBS -l nodes=1:ppn={cpu_count}:gpus={args.hpc_gpu_count}{shared_setting}{feature}\n')
                    else:
                        fp.write(f'#PBS -l nodes=1:ppn={cpu_count}\n')

                    fp.write(f'#PBS -l mem={cpu_count * 5}gb\n') #hpc limit

                    walltime = 96
                    if args.hpc_queue == 'fast':
                        walltime = 8
                    elif args.hpc_queue == 'inf':
                        walltime = 144
                    fp.write(f'#PBS -l walltime={walltime}:00:00\n\n')
                    fp.write(f'module load conda\n')

                    fp.write(f'export TMPDIR=$HOME/tmp\n')
                    fp.write(f'export TEMP=$HOME/tmp\n')

                    fp.write(f'export SDL_AUDIODRIVER=waveout\n')
                    fp.write(f'export SDL_VIDEODRIVER=x11\n')

                    fp.write(f'cd {path_base}\n')
                else:
                    if not is_windows:
                        fp.write(f'#!/bin/bash -v\n')

                if is_windows:
                    fp.write(f'CALL activate {args.conda_env}\n')
                else:
                    fp.write(f'source activate {args.conda_env}\n')

            if args.is_xvfb:
                fp.write(f'Xvfb :{xvfb_disp_id} -screen 0 1024x768x24 </dev/null &\n')
                fp.write(f'export DISPLAY=":{xvfb_disp_id}"\n')

            # each task
            if is_windows:
                marker_file_part = ''
                if max_process_per_task > 1:
                    name_tmp = params_comb['name']
                    log_tmp = f'{path_base}/tasks/{args.report}/{name_tmp}.log'
                    marker_file_part = f'^> {log_tmp}'
                    windows_log_list.append(marker_file_part)

                fp.write(f'start cmd /c python {path_base}/{args.main_script} {str_params} {each_task_line_end} {marker_file_part}\n')
            else:
                fp.write(f'python {path_base}/{args.main_script} {str_params} {each_task_line_end}\n')

            # last line
            if process_per_task + 1 == max_process_per_task and max_process_per_task > 1:
                if is_windows:
                    fp.write(f':wait\n')
                    for each in windows_log_list:
                        fp.write(f'if not exist {each} goto wait\n')
                else:
                    fp.write(f'wait\n')

        process_per_task += 1
        if process_per_task == max_process_per_task or \
                (idx_comb == len(grid)-1 and idx_repeat == args.repeat-1):
            process_per_task = 0
            windows_log_list = []

            if not is_windows:
                cmd = f'chmod +x {script_path}'
                logging.info(cmd)
                stdout = subprocess.check_output(cmd, shell=True, encoding='utf-8')
                logging.info(stdout)
            cmd = script_path

            if args.is_hpc:
                cmd = 'qsub -N ' + params_comb['name'] + ' ' + script_path

            any_count += 1
            logging.info('\n\n\n\nTask: {} / {}'.format(any_count, max_count))
            logging.info(cmd)
            stdout = subprocess.check_output(cmd, shell=True, encoding='utf-8')
            logging.info(stdout)

            if args.single_task:
                logging.info('Single task test mode completed')
                exit()

        xvfb_disp_id += 1
        if xvfb_disp_id > 90:
            xvfb_disp_id = 1