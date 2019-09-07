"""
An example of Adversarial Inverse Reinforcement Learning.
"""

import argparse
import json
import os
from pprint import pprint
import pickle

import numpy as np
import torch
import gym

from machina.pols import GaussianPol, CategoricalPol, MultiCategoricalPol
from machina.algos import airl, behavior_clone
from machina.vfuncs import DeterministicSVfunc, DeterministicSAVfunc
from machina.envs import GymEnv, C2DEnv
from machina.traj import Traj
from machina.traj import epi_functional as ef
from machina.samplers import EpiSampler
from machina import logger
from machina.utils import measure, set_device

from util.simple_net import PolNet, PolNetLSTM, VNet, DiscrimNet, VNetLSTM

import premaidai_gym

parser = argparse.ArgumentParser()
parser.add_argument('--log', type=str, default='garbage_airl',
                    help='Directory name of log.')
parser.add_argument('--env_name', type=str,
                    default='RoboschoolPremaidAIWalker-v0', help='Name of environment.')
                    # default='Pendulum-v0', help='Name of environment.')
parser.add_argument('--c2d', action='store_true',
                    default=False, help='If True, action is discretized.')
parser.add_argument('--record', action='store_true',
                    default=True, help='If True, movie is saved.')
                    # default=False, help='If True, movie is saved.')
parser.add_argument('--seed', type=int, default=256)
parser.add_argument('--max_epis', type=int,
                    default=100000000, help='Number of episodes to run.')
parser.add_argument('--num_parallel', type=int, default=4,
                    help='Number of processes to sample.')
parser.add_argument('--cuda', type=int, default=-1, help='cuda device number.')
parser.add_argument('--data_parallel', action='store_true', default=False,
                    help='If True, inference is done in parallel on gpus.')

parser.add_argument('--expert_dir', type=str, default='data/expert_epis',
                    help='Directory path storing file of expert trajectory.')
parser.add_argument('--expert_fname', type=str,
                    default='RoboschoolPremaidAIWalker-v0_100epis.pkl', help='Name of pkl file of expert trajectory')
                    # default='Pendulum-v0_100epis.pkl', help='Name of pkl file of expert trajectory')

parser.add_argument('--max_steps_per_iter', type=int, default=50000,
                    help='Number of steps to use in an iteration.')
parser.add_argument('--batch_size', type=int, default=50000)
parser.add_argument('--discrim_batch_size', type=int, default=32)
parser.add_argument('--pol_lr', type=float, default=1e-4,
                    help='Policy learning rate.')
parser.add_argument('--vf_lr', type=float, default=1e-3,
                    help='Value function learning rate.')
parser.add_argument('--discrim_lr', type=float, default=3e-4,
                    help='Discriminator learning rate.')

parser.add_argument('--epoch_per_iter', type=int, default=50,
                    help='Number of epoch in an iteration')
parser.add_argument('--discrim_step', type=int, default=10)

parser.add_argument('--gamma', type=float, default=0.995,
                    help='Discount factor.')
parser.add_argument('--lam', type=float, default=0.97,
                    help='Tradeoff value of bias variance.')
parser.add_argument('--pol_ent_beta', type=float, default=0,
                    help='Entropy coefficient for policy.')

parser.add_argument('--max_grad_norm', type=float, default=10,
                    help='Value of maximum gradient norm.')

parser.add_argument('--pol_h1', type=int, default=100,
                    help='Hidden size of layer1 of policy.')
parser.add_argument('--pol_h2', type=int, default=100,
                    help='Hidden size of layer2 of policy.')
parser.add_argument('--vf_h1', type=int, default=32,
                    help='Hidden size of layer1 of value function.')
parser.add_argument('--vf_h2', type=int, default=32,
                    help='Hidden size of layer2 of value function.')
parser.add_argument('--discrim_h1', type=int, default=100,
                    help='Hidden size of layer1 of discriminator.')
parser.add_argument('--discrim_h2', type=int, default=100,
                    help='Hidden size of layer2 of discriminator.')

parser.add_argument('--rl_type', type=str,
                    choices=['trpo', 'ppo_clip', 'ppo_kl'], default='trpo', help='Choice for Reinforcement Learning algorithms.')

parser.add_argument('--clip_param', type=float, default=0.2,
                    help='Value of clipping liklihood ratio.')
parser.add_argument('--kl_targ', type=float, default=0.01,
                    help='Target value of kl divergence.')
parser.add_argument('--init_kl_beta', type=float,
                    default=1, help='Initial kl coefficient.')

parser.add_argument('--pretrain', action='store_true', default=True,
                    help='If True, policy is pretrained by behavioral cloning.')
parser.add_argument('--bc_batch_size', type=int, default=256)
parser.add_argument('--bc_epoch', type=int, default=1000)

parser.add_argument('--rew_type', type=str,
                    choices=['adv', 'rew'], default='rew', help='Choice for reward type.')
args = parser.parse_args()

if not os.path.exists(args.log):
    os.mkdir(args.log)

with open(os.path.join(args.log, 'args.json'), 'w') as f:
    json.dump(vars(args), f)
pprint(vars(args))

if not os.path.exists(os.path.join(args.log, 'models')):
    os.mkdir(os.path.join(args.log, 'models'))

np.random.seed(args.seed)
torch.manual_seed(args.seed)

device_name = 'cpu' if args.cuda < 0 or args.rl_type == 'trpo' else "cuda:{}".format(
    args.cuda)
device = torch.device(device_name)
set_device(device)

score_file = os.path.join(args.log, 'progress.csv')
logger.add_tabular_output(score_file)

env = GymEnv(args.env_name, log_dir=os.path.join(
    args.log, 'movie'), record_video=args.record)
env.env.seed(args.seed)
if args.c2d:
    env = C2DEnv(env)

observation_space = env.observation_space
action_space = env.action_space


pol_net = PolNet(observation_space, action_space)
if isinstance(action_space, gym.spaces.Box):
    pol = GaussianPol(observation_space, action_space, pol_net,
                      data_parallel=args.data_parallel)
elif isinstance(action_space, gym.spaces.Discrete):
    pol = CategoricalPol(observation_space, action_space, pol_net,
                         data_parallel=args.data_parallel)
elif isinstance(action_space, gym.spaces.MultiDiscrete):
    pol = MultiCategoricalPol(
        observation_space, action_space, pol_net, data_parallel=args.data_parallel)
else:
    raise ValueError('Only Box, Discrete, and MultiDiscrete are supported')

vf_net = VNet(observation_space)
vf = DeterministicSVfunc(observation_space, vf_net,
                         data_parallel=args.data_parallel)

sampler = EpiSampler(env, pol, num_parallel=args.num_parallel, seed=args.seed)

optim_pol = torch.optim.Adam(pol_net.parameters(), args.pol_lr)
optim_vf = torch.optim.Adam(vf_net.parameters(), args.vf_lr)

with open(os.path.join(args.expert_dir, args.expert_fname), 'rb') as f:
    expert_epis = pickle.load(f)
expert_traj = Traj()
expert_traj.add_epis(expert_epis)
expert_traj = ef.add_next_obs(expert_traj)
expert_traj.register_epis()
expert_rewards = [np.sum(epi['rews']) for epi in expert_epis]
expert_mean_rew = np.mean(expert_rewards)
logger.log('expert_score={}'.format(expert_mean_rew))
logger.log('expert_num_epi={}'.format(expert_traj.num_epi))


total_epi = 0
total_step = 0
max_rew = -1e6
kl_beta = args.init_kl_beta

if args.pretrain:
    with measure('bc pretrain'):
        for _ in range(args.bc_epoch):
            _ = behavior_clone.train(
                expert_traj, pol, optim_pol, args.bc_batch_size
            )

while args.max_epis > total_epi:
    with measure('sample'):
        epis = sampler.sample(pol, max_steps=args.max_steps_per_iter)
    with measure('train'):
        traj = Traj()
        traj.add_epis(epis)

        traj = ef.compute_vs(traj, vf)
        traj = ef.compute_rets(traj, args.gamma)
        traj = ef.compute_advs(traj, args.gamma, args.lam)
        traj = ef.centerize_advs(traj)
        traj = ef.compute_h_masks(traj)
        traj.register_epis()

        if args.data_parallel:
            pol.dp_run = True
            vf.dp_run = True

        if args.ppo_type == 'clip':
            result_dict = ppo_clip.train(traj=traj, pol=pol, vf=vf, clip_param=args.clip_param,
                                         optim_pol=optim_pol, optim_vf=optim_vf, epoch=args.epoch_per_iter, batch_size=args.batch_size if not args.rnn else args.rnn_batch_size, max_grad_norm=args.max_grad_norm)
        else:
            result_dict = ppo_kl.train(traj=traj, pol=pol, vf=vf, kl_beta=kl_beta, kl_targ=args.kl_targ,
                                       optim_pol=optim_pol, optim_vf=optim_vf, epoch=args.epoch_per_iter, batch_size=args.batch_size if not args.rnn else args.rnn_batch_size, max_grad_norm=args.max_grad_norm)
            kl_beta = result_dict['new_kl_beta']

        if args.data_parallel:
            pol.dp_run = False
            vf.dp_run = False

    total_epi += traj.num_epi
    step = traj.num_step
    total_step += step
    rewards = [np.sum(epi['rews']) for epi in epis]
    mean_rew = np.mean(rewards)
    logger.record_results(args.log, result_dict, score_file,
                          total_epi, step, total_step,
                          rewards,
                          plot_title=args.env_name)

    if mean_rew > max_rew:
        torch.save(pol.state_dict(), os.path.join(
            args.log, 'models', 'pol_max.pkl'))
        torch.save(vf.state_dict(), os.path.join(
            args.log, 'models', 'vf_max.pkl'))
        torch.save(optim_pol.state_dict(), os.path.join(
            args.log, 'models', 'optim_pol_max.pkl'))
        torch.save(optim_vf.state_dict(), os.path.join(
            args.log, 'models', 'optim_vf_max.pkl'))
        max_rew = mean_rew

    torch.save(pol.state_dict(), os.path.join(
        args.log, 'models', 'pol_last.pkl'))
    torch.save(vf.state_dict(), os.path.join(
        args.log, 'models', 'vf_last.pkl'))
    torch.save(optim_pol.state_dict(), os.path.join(
        args.log, 'models', 'optim_pol_last.pkl'))
    torch.save(optim_vf.state_dict(), os.path.join(
        args.log, 'models', 'optim_vf_last.pkl'))
    del traj
del sampler
