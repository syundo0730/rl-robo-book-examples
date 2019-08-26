import time
import os

import numpy as np
import gym
import torch
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


# define your environment
env_name = 'RoboschoolPremaidAIWalker-v0'
env = gym.make(env_name)
obs = env.reset()

# check dimension of observation space and action space
observation_space = env.observation_space
action_space = env.action_space

# define your policy
# policy
pol_net = PolNet(observation_space, action_space)
# pol = CategoricalPol(observation_space, action_space, pol_net)
pol = GaussianPol(observation_space, action_space, pol_net)
# value function
vf_net = VNet(observation_space)
vf = DeterministicSVfunc(observation_space, vf_net)

# set optimizer to both models
optim_pol = torch.optim.Adam(pol_net.parameters(), lr=1e-4)
optim_vf = torch.optim.Adam(vf_net.parameters(), lr=3e-4)

#  arguments of PPO
gamma = 0.995
lam = 1
clip_param = 0.2
epoch_per_iter = 50
batch_size = 64
max_grad_norm = 10


sampler = EpiSampler(env, pol, num_parallel=2, seed=42)

# machina automatically write log (model ,scores, etc..)
log_dir_name = 'garbage'
if not os.path.exists(log_dir_name):
    os.mkdir(log_dir_name)
    os.mkdir(log_dir_name+'/models')
score_file = os.path.join(log_dir_name, 'progress.csv')
logger.add_tabular_output(score_file)

# counter and record for loop
total_epi = 0
total_step = 0
max_rew = -500

# how long will you train
max_episodes = 100  # for100 eposode

# max timesteps per eposode
max_steps_per_iter = 150  # 150 frames (= 10 sec)

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
    del traj
del sampler


# load best policy
best_path = 'garbage/models/pol_max.pkl'
best_pol = GaussianPol(observation_space, action_space, pol_net)
best_pol.load_state_dict(torch.load(best_path))

# show your trained policy's behavior
done = False
o = env.reset()
for _ in range(300):  # show 300 frames (=20 sec)
    if done:
        time.sleep(1)  # when the boundary　of eposode
        o = env.reset()
    ac_real, ac, a_i = best_pol.deterministic_ac_real(torch.tensor(o, dtype=torch.float))
    ac_real = ac_real.reshape(pol.action_space.shape)
    next_o, r, done, e_i = env.step(np.array(ac_real))
    o = next_o
    time.sleep(1/15)  # 15fps
    env.render()

env.close()
