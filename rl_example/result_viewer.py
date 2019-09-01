import time

import gym
import numpy as np
import torch
from machina.pols import GaussianPol
from machina.vfuncs import DeterministicSVfunc

from util.simple_net import PolNet, VNet


seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

log_dir_name = 'garbage'
env_name = 'RoboschoolPremaidAIWalker-v0'
env = gym.make(env_name)
env.seed(seed)

# check dimension of observation space and action space
observation_space = env.observation_space
action_space = env.action_space

# policy
pol_net = PolNet(observation_space, action_space)
pol = GaussianPol(observation_space, action_space, pol_net)
# value function
vf_net = VNet(observation_space)
vf = DeterministicSVfunc(observation_space, vf_net)

# load best policy
best_path = f'{log_dir_name}/models/pol_max.pkl'
best_pol = GaussianPol(observation_space, action_space, pol_net)
best_pol.load_state_dict(torch.load(best_path))

# show trained policy's behavior
done = False
o = env.reset()
for _ in range(1000):  # show 16.5 sec (0.0165 * 1000)
    if done:
        time.sleep(1)  # when the boundary　of eposode
        o = env.reset()
    ac_real, ac, a_i = best_pol.deterministic_ac_real(torch.tensor(o, dtype=torch.float))
    ac_real = ac_real.reshape(pol.action_space.shape)
    next_o, r, done, e_i = env.step(np.array(ac_real))
    o = next_o
    env.render()
    time.sleep(0.0165)
env.close()
