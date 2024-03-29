"""
An example of Generative Adversarial Imitation Learning.
"""

import argparse
import copy
import json
import os
from pprint import pprint
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym

import machina as mc
from machina.pols import GaussianPol, CategoricalPol, MultiCategoricalPol
from machina.algos import gail, behavior_clone
from machina.vfuncs import DeterministicSVfunc, DeterministicSAVfunc
from machina.envs import GymEnv, C2DEnv
from machina.traj import Traj
from machina.traj import epi_functional as ef
from machina.samplers import EpiSampler
from machina import logger
from machina.utils import measure, set_device

from util.simple_net import PolNet, PolNetLSTM, VNet, VNetLSTM, DiscrimNet

import premaidai_gym

parser = argparse.ArgumentParser()
parser.add_argument('--log', type=str, default='garbage_gail',
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
parser.add_argument('--pol_lr', type=float, default=1e-4)
parser.add_argument('--vf_lr', type=float, default=1e-3)
parser.add_argument('--discrim_lr', type=float, default=3e-4)

parser.add_argument('--epoch_per_iter', type=int, default=50,
                    help='Number of epoch in an iteration')
parser.add_argument('--discrim_step', type=int, default=10)

parser.add_argument('--gamma', type=float, default=0.995,
                    help='Discount factor.')
parser.add_argument('--lam', type=float, default=0.97,
                    help='Tradeoff value of bias variance.')
parser.add_argument('--pol_ent_beta', type=float, default=0,
                    help='Entropy coefficient for policy.')
parser.add_argument('--discrim_ent_beta', type=float, default=0,
                    help='Entropy coefficient for discriminator.')

parser.add_argument('--rnn', action='store_true',
                    default=True, help='If True, network is reccurent.')
                    # default=False, help='If True, network is reccurent.')
parser.add_argument('--rnn_batch_size', type=int, default=8,
                    help='Number of sequences included in batch of rnn.')
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
                    choices=['trpo', 'ppo_clip', 'ppo_kl'], default='trpo')
parser.add_argument('--clip_param', type=float, default=0.2)
parser.add_argument('--kl_targ', type=float, default=0.01)
parser.add_argument('--init_kl_beta', type=float, default=1)

parser.add_argument('--bc_batch_size', type=int, default=256)
parser.add_argument('--pretrain', action='store_true', default=False)
parser.add_argument('--bc_epoch', type=int, default=1000)
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

if args.rnn:
    pol_net = PolNetLSTM(observation_space, action_space,
                         h_size=256, cell_size=256)
else:
    pol_net = PolNet(observation_space, action_space)
if isinstance(action_space, gym.spaces.Box):
    pol = GaussianPol(observation_space, action_space, pol_net, args.rnn,
                      data_parallel=args.data_parallel, parallel_dim=1 if args.rnn else 0)
elif isinstance(action_space, gym.spaces.Discrete):
    pol = CategoricalPol(observation_space, action_space, pol_net, args.rnn,
                         data_parallel=args.data_parallel, parallel_dim=1 if args.rnn else 0)
elif isinstance(action_space, gym.spaces.MultiDiscrete):
    pol = MultiCategoricalPol(observation_space, action_space, pol_net, args.rnn,
                              data_parallel=args.data_parallel, parallel_dim=1 if args.rnn else 0)
else:
    raise ValueError('Only Box, Discrete, and MultiDiscrete are supported')

if args.rnn:
    vf_net = VNetLSTM(observation_space, h_size=256, cell_size=256)
else:
    vf_net = VNet(observation_space)
vf = DeterministicSVfunc(observation_space, vf_net, args.rnn,
                         data_parallel=args.data_parallel, parallel_dim=1 if args.rnn else 0)


discrim_net = DiscrimNet(
    observation_space, action_space, h1=args.discrim_h1, h2=args.discrim_h2)
discrim = DeterministicSAVfunc(
    observation_space, action_space, discrim_net, data_parallel=args.data_parallel)

sampler = EpiSampler(env, pol, num_parallel=args.num_parallel, seed=args.seed)

optim_pol = torch.optim.Adam(pol_net.parameters(), args.pol_lr)
optim_vf = torch.optim.Adam(vf_net.parameters(), args.vf_lr)
optim_discrim = torch.optim.Adam(discrim_net.parameters(), args.discrim_lr)

with open(os.path.join(args.expert_dir, args.expert_fname), 'rb') as f:
    expert_epis = pickle.load(f)
expert_traj = Traj()
expert_traj.add_epis(expert_epis)
expert_traj.register_epis()
expert_rewards = [np.sum(epi['rews']) for epi in expert_epis]
expert_mean_rew = np.mean(expert_rewards)
logger.log('expert_score={}'.format(expert_mean_rew))
logger.log('expert_num_epi={}'.format(expert_traj.num_epi))


total_epi = 0
total_step = 0
max_rew = -1e6

if args.rl_type == 'ppo_kl':
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
        agent_traj = Traj()
        agent_traj.add_epis(epis)
        agent_traj = ef.compute_pseudo_rews(agent_traj, discrim)
        agent_traj = ef.compute_vs(agent_traj, vf)
        agent_traj = ef.compute_rets(agent_traj, args.gamma)
        agent_traj = ef.compute_advs(agent_traj, args.gamma, args.lam)
        agent_traj = ef.centerize_advs(agent_traj)
        agent_traj = ef.compute_h_masks(agent_traj)
        agent_traj.register_epis()

        if args.data_parallel:
            pol.dp_run = True
            vf.dp_run = True
            discrim.dp_run = True

        if args.rl_type == 'trpo':
            result_dict = gail.train(agent_traj, expert_traj, pol, vf, discrim, optim_vf, optim_discrim,
                                     rl_type=args.rl_type,
                                     epoch=args.epoch_per_iter,
                                     batch_size=args.batch_size if not args.rnn else args.rnn_batch_size, discrim_batch_size=args.discrim_batch_size,
                                     discrim_step=args.discrim_step,
                                     pol_ent_beta=args.pol_ent_beta, discrim_ent_beta=args.discrim_ent_beta)
        elif args.rl_type == 'ppo_clip':
            result_dict = gail.train(agent_traj, expert_traj, pol, vf, discrim, optim_vf, optim_discrim,
                                     rl_type=args.rl_type,
                                     epoch=args.epoch_per_iter,
                                     batch_size=args.batch_size if not args.rnn else args.rnn_batch_size,
                                     discrim_batch_size=args.discrim_batch_size,
                                     discrim_step=args.discrim_step,
                                     pol_ent_beta=args.pol_ent_beta, discrim_ent_beta=args.discrim_ent_beta,
                                     optim_pol=optim_pol,
                                     clip_param=args.clip_param, max_grad_norm=args.max_grad_norm)

        else:
            result_dict = gail.train(agent_traj, expert_traj, pol, vf, discrim, optim_vf, optim_discrim,
                                     rl_type=args.rl_type,
                                     pol_ent_beta=args.pol_ent_beta, discrim_ent_beta=args.discrim_ent_beta,
                                     epoch=args.epoch_per_iter,
                                     batch_size=args.batch_size if not args.rnn else args.rnn_batch_size,
                                     discrim_batch_size=args.discrim_batch_size,
                                     discrim_step=args.discrim_step,
                                     optim_pol=optim_pol,
                                     kl_beta=kl_beta, kl_targ=args.kl_targ, max_grad_norm=args.max_grad_norm)
            kl_beta = result_dict['new_kl_beta']

        if args.data_parallel:
            pol.dp_run = False
            vf.dp_run = False
            discrim.dp_run = False

    total_epi += agent_traj.num_epi
    step = agent_traj.num_step
    total_step += step
    rewards = [np.sum(epi['real_rews']) for epi in epis]
    mean_rew = np.mean(rewards)
    pseudo_rewards = [np.sum(epi['rews']) for epi in epis]
    result_dict['PseudoReward'] = pseudo_rewards
    logger.record_results(args.log, result_dict, score_file,
                          total_epi, step, total_step,
                          rewards,
                          plot_title=args.env_name)

    mean_rew = np.mean([np.sum(path['real_rews']) for path in epis])
    if mean_rew > max_rew:
        torch.save(pol.state_dict(), os.path.join(
            args.log, 'models', 'pol_max.pkl'))
        torch.save(vf.state_dict(), os.path.join(
            args.log, 'models', 'vf_max.pkl'))
        torch.save(discrim.state_dict(), os.path.join(
            args.log, 'models', 'discrim_max.pkl'))
        torch.save(optim_pol.state_dict(), os.path.join(
            args.log, 'models', 'optim_pol_max.pkl'))
        torch.save(optim_vf.state_dict(), os.path.join(
            args.log, 'models', 'optim_vf_max.pkl'))
        torch.save(optim_discrim.state_dict(), os.path.join(
            args.log, 'models', 'optim_discrim_max.pkl'))
        max_rew = mean_rew

    torch.save(pol.state_dict(), os.path.join(
        args.log, 'models', 'pol_last.pkl'))
    torch.save(vf.state_dict(), os.path.join(
        args.log, 'models', 'vf_last.pkl'))
    torch.save(discrim.state_dict(), os.path.join(
        args.log, 'models', 'discrim_last.pkl'))
    torch.save(optim_pol.state_dict(), os.path.join(
        args.log, 'models', 'optim_pol_last.pkl'))
    torch.save(optim_vf.state_dict(), os.path.join(
        args.log, 'models', 'optim_vf_last.pkl'))
    torch.save(optim_discrim.state_dict(), os.path.join(
        args.log, 'models', 'optim_discrim_last.pkl'))
    del agent_traj
del sampler
