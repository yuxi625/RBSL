import argparse
import numpy as np
import gym
import panda_gym

import os
import csv
# from envs.push_obstacle import FetchPushObstacleEnv
from mpi4py import MPI
from envs import register_envs
from envs.multi_world_wrapper import NoisyAction
from rl_modules.cactionablemodel_agent import ActionableModel
from rl_modules.gofar_agent import GoFAR
from rl_modules.gcsl_agent import GCSL
from rl_modules.rbsl_agent import RBSL

import random
import torch
import wandb

"""
train the agent, the MPI part code is copy from openai baselines(https://github.com/openai/baselines/blob/master/baselines/her)

"""


def boolean(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args():
    # 创建了一个 ArgumentParser 对象，它是用来解析命令行输入参数的工具。
    # ArgumentParser 类是 Python 标准库 argparse 模块的一部分。
    parser = argparse.ArgumentParser()
    # the environment setting
    parser.add_argument('--env', type=str, default='FetchPushObstacle', help='the environment name')
    parser.add_argument('--n-epochs', type=int, default=500, help='the number of epochs to train the agent')
    parser.add_argument('--n-cycles', type=int, default=20, help='the times to collect samples per epoch')
    parser.add_argument('--n-batches', type=int, default=20, help='the times to update the network')
    parser.add_argument('--n-safety-batches', type=int, default=30, help='the times to update the safety network')
    parser.add_argument('--save-interval', type=int, default=5, help='the interval that save the trajectory')
    parser.add_argument('--num-workers', type=int, default=1, help='the number of cpus to collect samples')
    parser.add_argument('--replay-strategy', type=str, default='future', help='the HER strategy')
    parser.add_argument('--safety-replay-strategy', type=str, default='recovery', help='the recovery strategy')
    parser.add_argument('--clip-return', type=float, default=50, help='if clip the returns')
    parser.add_argument('--save-dir', type=str, default='saved_models/', help='the path to save the models')
    parser.add_argument('--random-eps', type=float, default=0.3, help='random eps')
    parser.add_argument('--buffer-size', type=int, default=int(2e6), help='the size of the buffer')
    parser.add_argument('--safety-buffer-size', type=int, default=int(2e6), help='the size of the safety buffer')
    parser.add_argument('--replay-k', type=int, default=4, help='ratio to be replace')
    parser.add_argument('--clip-obs', type=float, default=200, help='the clip ratio')
    parser.add_argument('--batch-size', type=int, default=512, help='the sample batch size')
    parser.add_argument('--gamma', type=float, default=0.98, help='the discount factor')
    parser.add_argument('--action-l2', type=float, default=1, help='l2 reg')
    parser.add_argument('--lr-actor', type=float, default=0.001, help='the learning rate of the actor')
    parser.add_argument('--lr-critic', type=float, default=0.001, help='the learning rate of the critic')
    parser.add_argument('--lr-safety-actor', type=float, default=0.001, help='the learning rate of the actor')
    parser.add_argument('--lr-safety-critic', type=float, default=0.001, help='the learning rate of the critic')
    parser.add_argument('--lr-scalar', type=float, default=5e-4, help='the learning rate of the scalar')
    parser.add_argument('--polyak', type=float, default=0.95, help='the average coefficient')
    parser.add_argument('--n-test-rollouts', type=int, default=60, help='the number of tests')
    parser.add_argument('--clip-range', type=float, default=5, help='the clip range')
    parser.add_argument('--demo-length', type=int, default=20, help='the demo length')
    parser.add_argument('--cuda', default=True, type=boolean, help='if use gpu do the acceleration')
    parser.add_argument('--device', default=0, type=int, help='gpu device number')
    parser.add_argument('--num-rollouts-per-mpi', type=int, default=1, help='the rollouts per mpi')

    # hyperparameters that need to be changed
    parser.add_argument('--eval', default=True, type=boolean)
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--method', default='rbsl', type=str)
    parser.add_argument('--f', default='chi', type=str)
    parser.add_argument('--online', default=False, type=boolean)

    parser.add_argument('--noise', default=False, type=boolean, help='add noise to action')
    parser.add_argument('--noise-eps', type=float, default=1.0, help='noise eps')

    parser.add_argument('--relabel', default=True, type=boolean)
    parser.add_argument('--relabel_percent', default=1.0, type=float)
    parser.add_argument('--safety_relabel_percent', default=0.5, type=float)

    parser.add_argument('--mutip', default=0.01, type=float)

    parser.add_argument('--reward_type', default='binary', type=str)
    parser.add_argument('--threshold', default=0.05, type=float)
    parser.add_argument('--disc_iter', type=int, default=20)
    parser.add_argument('--disc_lambda', type=float, default=0.01)

    parser.add_argument('--expert_percent', type=float, default=1, help='the expert coefficient')
    parser.add_argument('--random_percent', type=float, default=0, help='the random coefficient')

    args = parser.parse_args()
    #代码调用了 parse_args() 方法，该方法会解析命令行输入的参数，并将其存储在一个 Namespace 对象中。
    # 这个对象的属性就是命令行参数的名称，对应的值就是命令行输入的值。
    # 例如，如果命令行输入 --env FetchReachObstacle, 那么 args.env 的值就会是 'FetchReachObstacle'。

    return args



def get_env_params(env):
    obs = env.reset()
    # close the environment
    params = {'obs': obs['observation'].shape[0],
            'goal': obs['desired_goal'].shape[0],
            'action': env.action_space.shape[0],
            'action_max': env.action_space.high[0],
            'action_space': env.action_space
            }
    params['max_timesteps'] = env._max_episode_steps
    return params

def get_full_envname(name):
    dic = {
        'FetchPushObstacle':'FetchPushObstacle-v0',
        'FetchPickAndPlaceObstacle':'FetchPickAndPlaceObstacle-v0',
        'FetchSlideObstacle':'FetchSlideObstacle-v0',
        'FetchReachObstacle':'FetchReachObstacle-v0',
        'PandaPush':'PandaPush-v2'
    }
    if name in dic.keys():
        return dic[name]
    else:
        return name

def get_method_params(args):
    if args.online:
        args.n_batches = 40
        args.n_cycles = 50
        
    if args.method == 'AMlag':
        args.lr_actor = 0.001
        args.lr_critic = 0.001
    else:
        args.lr_actor = 5e-4
        args.lr_critic = 5e-4

    if 'gcsl' in args.method:
        args.relabel_percent = 1.0
    if args.method == 'rbsl':
        if (args.env == 'FetchPushObstacle' or args.env == 'FetchSlideObstacle' or args.env == 'PandaPush' or args.env == 'FetchPickAndPlaceObstacle') and (args.expert_percent == 0.5 or args.expert_percent == 1):
            args.relabel_percent = 0
        else:
            args.relabel_percent = 1.0
    if 'AMlag' in args.method: 
        args.relabel_percent = 0.5
    if 'gcbc' in args.method:
        args.relabel = False 
    if 'gofar' in args.method:
        args.relabel = False 
        args.reward_type = 'disc' 


def launch(args):

    # 根据不同的训练方法（args.method）来调整超参数
    get_method_params(args)
    # 设置了一个新的参数 args.use_disc，用于判断是否使用离散奖励函数
    args.use_disc = True if args.reward_type =='disc' else False
    # 获得完整环境名称
    args.env_id = get_full_envname(args.env)

    # 加载环境
    env = gym.make(args.env_id)

    # stochastic environment setting
    if args.noise:
        # NoisyAction 类的作用是在每个环境步骤中向环境的动作添加噪声，noise_eps 参数控制噪声的强度。
        # 这样做可以增加环境的随机性，使得训练的策略更具鲁棒性。
        env = NoisyAction(env, noise_eps=args.noise_eps)
        # 将最大步数（episode step）设置为 50
        env._max_episode_steps = 50

    # relabel_percent 参数表示在训练过程中，有多少比例的动作会被重新标记（relabel），
    # 以便进行HER（Hindsight Experience Replay）训练。
    # relabel 为 False 时，自然不会有动作被重新标记，因此这个比例被设置为 0。
    if args.relabel == False:
        args.relabel_percent = 0.
    # 字符串的格式为 'relabelX'，其中 X 是 relabel_percent 的值。
    # 如果 relabel_percent 为 0.5，则 relabel_tag 的值为 'relabel0.5'。
    # 这个标签会在后续生成的训练运行名称（run_name）中被使用，从而帮助识别不同训练条件下的训练结果。
    relabel_tag = f'relabel{args.relabel_percent}'

    # reward function
    reward_tag = args.reward_type

    if args.reward_type == 'disc':
        reward_tag = f'{args.disc_iter}disc{args.disc_lambda}'
    elif args.reward_type == 'binary':
        reward_tag = f'{args.reward_type}{args.threshold}'
    run_name = f'{args.env_id}-{args.expert_percent}-{args.random_percent}-{args.method}-{reward_tag}-{relabel_tag}-{args.seed}'

    if args.noise:
        run_name = f'{args.env_id}-noise{args.noise_eps}-{args.expert_percent}-{args.random_percent}-{args.method}-{reward_tag}-{relabel_tag}-{args.seed}'

    args.run_name = run_name

    if MPI.COMM_WORLD.Get_rank() == 0:
        wandb.init(project='data', name=run_name, 
        group=args.env, config=args)

    # set random seeds for reproduce
    env.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    np.random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    torch.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())
    if args.cuda:
        torch.cuda.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())
    
    # get the environment parameters
    env_params = get_env_params(env)

    # create agent 用于创建强化学习训练器
    if args.method == 'gofar':
        trainer = GoFAR(args, env, env_params)
    elif 'gcsl' in args.method or 'gcbc' in args.method:
        trainer = GCSL(args, env, env_params)
    elif 'action' in args.method or 'AM' in args.method:
        trainer = ActionableModel(args, env, env_params)
    elif args.method == 'rbsl':
        trainer = RBSL(args, env, env_params)
    else:
        raise NotImplementedError

    # 运行名称包含了环境名称、专家系数、随机系数、训练方法、奖励类型、重新标记比例以及随机种子等信息。
    # 为了在终端或日志文件中记录当前训练的具体条件，方便后续查看和对比不同条件下训练的结果
    print(run_name)
    
    # do offline goal-conditioned rl
    # 启动训练 收集训练结果
    # evaluate_agent=args.eval 参数表示是否在训练过程中进行评估，以获取训练器在不同阶段的表现
    result = trainer.learn(evaluate_agent=args.eval)

    return result

if __name__ == '__main__':
    # take the configuration for the HER
    # 设置环境变量
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['IN_MPI'] = '1'
    os.environ['WANDB_MODE'] = 'offline'


    # 初始化列表，这些列表将用于计算多个种子训练结果的平均值和方差，以便评估算法的稳定性和性能。
    success_rates = []    #用于存储每次训练的成功率
    cost_returns = []            #用于存储每次训练的成本回报



    args = get_args()   #从命令行中获取用户设置的各项参数

    #一个包含两个不同随机种子值的集合
    seeds = {args.seed, args.seed + 1}

    # register_envs() 函数的主要功能（注册环境）是向 OpenAI Gym 的环境注册机制中注册自定义的环境。
    # 通过注册，这些自定义环境就可以像 Gym 内置的环境一样，通过 gym.make('环境ID') 来创建和使用。
    register_envs()

    # 遍历 seeds 集合，对每个种子值执行一次训练
    for seed in seeds:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)          # 指定GPU设备

        # run the launch
        result = launch(args)                                   # 执行单次训练
        success_rates.append(result[0])                      # 记录成功率
        cost_returns.append(result[1])                         # 记录成本回报



    mean_suc = sum(success_rates) / len(success_rates)    # 平均成功率
    mean_cost = sum(cost_returns) / len(cost_returns)       #平均成本
    var_suc = np.var(success_rates)                       # 方差成功率
    var_cost = np.var(cost_returns)                       # 方差成本

    #创建一个字典对象dict，用于存储和整理训练后的关键信息。
    # 这些信息包括使用的训练方法、训练的环境、训练结果的平均成功率、平均成本回报，以及成功率和成本回报的方差。
    # 通过整理这些信息，可以方便地对算法的性能和稳定性进行评估和分析。
    dict = {'method':f'{args.method}', 'task': f'{args.env}', 'suc':mean_suc, 'cost': mean_cost, 'sucv':var_suc, 'costv':var_cost}
    # print(dict)