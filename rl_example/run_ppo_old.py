import os
from math import inf

import numpy as np
import torch
from machina.envs import GymEnv
from machina.pols import GaussianPol
from machina.samplers import EpiSampler
from machina.vfuncs import DeterministicSVfunc
from machina.traj import epi_functional as ef
from machina import logger
from machina.utils import measure
from machina.traj import Traj
from machina.algos import ppo_clip
import premaidai_gym

from util.simple_net import PolNet, VNet


seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

log_dir_name = 'garbage'
env_name = 'RoboschoolPremaidAIWalker-v0'
env = GymEnv(env_name, log_dir=os.path.join(
    log_dir_name, 'movie'), record_video=True)
env.env.seed(seed)

# check dimension of observation space and action space
observation_space = env.observation_space
action_space = env.action_space

# policy
pol_net = PolNet(observation_space, action_space)
pol = GaussianPol(observation_space, action_space, pol_net)
# value function
vf_net = VNet(observation_space)
vf = DeterministicSVfunc(observation_space, vf_net)

# optimizer to both models
optim_pol = torch.optim.Adam(pol_net.parameters(), lr=1e-4)
optim_vf = torch.optim.Adam(vf_net.parameters(), lr=3e-4)

#  arguments of PPO
gamma = 0.99
lam = 0.95
clip_param = 0.2
epoch_per_iter = 4
batch_size = 64
max_grad_norm = 0.5
num_parallel = 16


sampler = EpiSampler(env, pol, num_parallel=num_parallel, seed=seed)

# machina automatically write log (model ,scores, etc..)
if not os.path.exists(log_dir_name):
    os.mkdir(log_dir_name)
if not os.path.exists(f'{log_dir_name}/models'):
    os.mkdir(f'{log_dir_name}/models')
score_file = os.path.join(log_dir_name, 'progress.csv')
logger.add_tabular_output(score_file)

# counter and record for loop
total_epi = 0
total_step = 0
max_rew = -inf
max_episodes = 1000000
max_steps_per_iter = 3000

# train loop
while max_episodes > total_epi:
    # sample trajectories
    with measure('sample'):
        epis = sampler.sample(pol, max_steps=max_steps_per_iter)

    # train from trajectories
    with measure('train'):
        traj = Traj()
        traj.add_epis(epis)

        # calulate advantage
        traj = ef.compute_vs(traj, vf)
        traj = ef.compute_rets(traj, gamma)
        traj = ef.compute_advs(traj, gamma, lam)
        traj = ef.centerize_advs(traj)
        traj = ef.compute_h_masks(traj)
        traj.register_epis()

        result_dict = ppo_clip.train(traj=traj, pol=pol, vf=vf, clip_param=clip_param,
                                     optim_pol=optim_pol, optim_vf=optim_vf,
                                     epoch=epoch_per_iter, batch_size=batch_size,
                                     max_grad_norm=max_grad_norm)
    # update counter and record
    total_epi += traj.num_epi
    step = traj.num_step
    total_step += step
    rewards = [np.sum(epi['rews']) for epi in epis]
    mean_rew = np.mean(rewards)
    logger.record_results(log_dir_name, result_dict, score_file,
                          total_epi, step, total_step,
                          rewards,
                          plot_title=env_name)
    if mean_rew > max_rew:
        torch.save(pol.state_dict(), os.path.join(
            log_dir_name, 'models', 'pol_max.pkl'))
        torch.save(vf.state_dict(), os.path.join(
            log_dir_name, 'models', 'vf_max.pkl'))
        torch.save(optim_pol.state_dict(), os.path.join(
            log_dir_name, 'models', 'optim_pol_max.pkl'))
        torch.save(optim_vf.state_dict(), os.path.join(
            log_dir_name, 'models', 'optim_vf_max.pkl'))
        max_rew = mean_rew

    torch.save(pol.state_dict(), os.path.join(
        log_dir_name, 'models', 'pol_last.pkl'))
    torch.save(vf.state_dict(), os.path.join(
        log_dir_name, 'models', 'vf_last.pkl'))
    torch.save(optim_pol.state_dict(), os.path.join(
        log_dir_name, 'models', 'optim_pol_last.pkl'))
    torch.save(optim_vf.state_dict(), os.path.join(
        log_dir_name, 'models', 'optim_vf_last.pkl'))
    del traj
del sampler
