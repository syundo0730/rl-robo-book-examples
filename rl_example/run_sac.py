"""
An example of Soft Actor Critic.
"""

import argparse
import json
import os
from pprint import pprint

import numpy as np
import torch
import torch.nn as nn

from machina.pols import GaussianPol
from machina.algos import sac
from machina.vfuncs import DeterministicSAVfunc
from machina.envs import GymEnv
from machina.traj import Traj
from machina.traj import epi_functional as ef
from machina.samplers import EpiSampler
from machina import logger
from machina.utils import set_device, measure

from util.simple_net import PolNet, QNet, VNet

import premaidai_gym


parser = argparse.ArgumentParser()
# parser.add_argument('--log', type=str, default='garbage',
parser.add_argument('--log', type=str, default='garbage_sac',
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
                    default=10000000, help='Number of episodes to run.')
                    # default=1000000, help='Number of episodes to run.')
parser.add_argument('--max_steps_off', type=int,
                    default=10000000000000, help='Number of steps stored in off traj.')
                    # default=1000000000000, help='Number of episodes stored in off traj.')
# parser.add_argument('--num_parallel', type=int, default=4,
parser.add_argument('--num_parallel', type=int, default=16,
                    help='Number of processes to sample.')
parser.add_argument('--cuda', type=int, default=-1, help='cuda device number.')
parser.add_argument('--data_parallel', action='store_true', default=False,
                    help='If True, inference is done in parallel on gpus.')

# parser.add_argument('--max_steps_per_iter', type=int, default=10000,
parser.add_argument('--max_steps_per_iter', type=int, default=100000,
                    help='Number of steps to use in an iteration.')
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--sampling', type=int, default=1,
                    help='Number of sampling in calculation of expectation.')
parser.add_argument('--no_reparam', action='store_true', default=False)
# parser.add_argument('--pol_lr', type=float, default=1e-4,
parser.add_argument('--pol_lr', type=float, default=3e-4,
                    help='Policy learning rate')
parser.add_argument('--qf_lr', type=float, default=3e-4,
                    help='Q function learning rate')

parser.add_argument('--ent_alpha', type=float, default=1,
                    help='Entropy coefficient.')
parser.add_argument('--tau', type=float, default=5e-3,
                    help='Coefficient of target function.')
parser.add_argument('--gamma', type=float, default=0.99,
                    help='Discount factor.')
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

device_name = 'cpu' if args.cuda < 0 else "cuda:{}".format(args.cuda)
device = torch.device(device_name)
set_device(device)

score_file = os.path.join(args.log, 'progress.csv')
logger.add_tabular_output(score_file)


class _RewScaledEnv(GymEnv):
    """GymEnv to scale reward for sac optimizer"""
    def __init__(self, scale):
        super().__init__(args.env_name, log_dir=os.path.join(args.log, 'movie'), record_video=args.record)
        self._scale = scale

    def step(self, action):
        next_obs, reward, done, info = super().step(action)
        return next_obs, self._scale * reward, done, info


env = _RewScaledEnv(scale=20)
env.env.seed(args.seed)

observation_space = env.observation_space
action_space = env.action_space

# pol_net = PolNet(observation_space, action_space)
pol_net = PolNet(observation_space, action_space, h1=256, h2=256)
pol = GaussianPol(observation_space, action_space, pol_net,
                  data_parallel=args.data_parallel, parallel_dim=0)

# qf_net1 = QNet(observation_space, action_space)
qf_net1 = QNet(observation_space, action_space, h1=256, h2=256)
qf1 = DeterministicSAVfunc(observation_space, action_space, qf_net1,
                           data_parallel=args.data_parallel, parallel_dim=0)
# targ_qf_net1 = QNet(observation_space, action_space)
targ_qf_net1 = QNet(observation_space, action_space, h1=256, h2=256)
targ_qf_net1.load_state_dict(qf_net1.state_dict())
targ_qf1 = DeterministicSAVfunc(
    observation_space, action_space, targ_qf_net1, data_parallel=args.data_parallel, parallel_dim=0)

# qf_net2 = QNet(observation_space, action_space)
qf_net2 = QNet(observation_space, action_space, h1=256, h2=256)
qf2 = DeterministicSAVfunc(observation_space, action_space, qf_net2,
                           data_parallel=args.data_parallel, parallel_dim=0)
# targ_qf_net2 = QNet(observation_space, action_space)
targ_qf_net2 = QNet(observation_space, action_space, h1=256, h2=256)
targ_qf_net2.load_state_dict(qf_net2.state_dict())
targ_qf2 = DeterministicSAVfunc(
    observation_space, action_space, targ_qf_net2, data_parallel=args.data_parallel, parallel_dim=0)

qfs = [qf1, qf2]
targ_qfs = [targ_qf1, targ_qf2]

log_alpha = nn.Parameter(torch.zeros((), device=device))

sampler = EpiSampler(env, pol, args.num_parallel, seed=args.seed)

optim_pol = torch.optim.Adam(pol_net.parameters(), args.pol_lr)
optim_qf1 = torch.optim.Adam(qf_net1.parameters(), args.qf_lr)
optim_qf2 = torch.optim.Adam(qf_net2.parameters(), args.qf_lr)
optim_qfs = [optim_qf1, optim_qf2]
optim_alpha = torch.optim.Adam([log_alpha], args.pol_lr)

off_traj = Traj(args.max_steps_off, traj_device='cpu')

total_epi = 0
total_step = 0
max_rew = -1e6

while args.max_epis > total_epi:
    with measure('sample'):
        epis = sampler.sample(pol, max_steps=args.max_steps_per_iter)

    with measure('train'):
        on_traj = Traj(traj_device='cpu')
        on_traj.add_epis(epis)

        on_traj = ef.add_next_obs(on_traj)
        on_traj.register_epis()

        off_traj.add_traj(on_traj)

        total_epi += on_traj.num_epi
        step = on_traj.num_step
        total_step += step

        if args.data_parallel:
            pol.dp_run = True
            for qf, targ_qf in zip(qfs, targ_qfs):
                qf.dp_run = True
                targ_qf.dp_run = True

        result_dict = sac.train(
            off_traj,
            pol, qfs, targ_qfs, log_alpha,
            optim_pol, optim_qfs, optim_alpha,
            step, args.batch_size,
            args.tau, args.gamma, args.sampling, not args.no_reparam
        )

        if args.data_parallel:
            pol.dp_run = False
            for qf, targ_qf in zip(qfs, targ_qfs):
                qf.dp_run = False
                targ_qf.dp_run = False

    rewards = [np.sum(epi['rews']) for epi in epis]
    mean_rew = np.mean(rewards)
    logger.record_results(args.log, result_dict, score_file,
                          total_epi, step, total_step,
                          rewards,
                          plot_title=args.env_name)

    if mean_rew > max_rew:
        torch.save(pol.state_dict(), os.path.join(
            args.log, 'models', 'pol_max.pkl'))
        torch.save(qf1.state_dict(), os.path.join(
            args.log, 'models', 'qf1_max.pkl'))
        torch.save(qf2.state_dict(), os.path.join(
            args.log, 'models', 'qf2_max.pkl'))
        torch.save(optim_pol.state_dict(), os.path.join(
            args.log, 'models', 'optim_pol_max.pkl'))
        torch.save(optim_qf1.state_dict(), os.path.join(
            args.log, 'models', 'optim_qf1_max.pkl'))
        torch.save(optim_qf2.state_dict(), os.path.join(
            args.log, 'models', 'optim_qf2_max.pkl'))
        max_rew = mean_rew

    torch.save(pol.state_dict(), os.path.join(
        args.log, 'models', 'pol_last.pkl'))
    torch.save(qf1.state_dict(), os.path.join(
        args.log, 'models', 'qf1_last.pkl'))
    torch.save(qf2.state_dict(), os.path.join(
        args.log, 'models', 'qf2_last.pkl'))
    torch.save(optim_pol.state_dict(), os.path.join(
        args.log, 'models', 'optim_pol_last.pkl'))
    torch.save(optim_qf1.state_dict(), os.path.join(
        args.log, 'models', 'optim_qf1_last.pkl'))
    torch.save(optim_qf2.state_dict(), os.path.join(
        args.log, 'models', 'optim_qf2_last.pkl'))
    del on_traj
del sampler
