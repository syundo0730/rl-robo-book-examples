import pickle
from math import radians, sin, pi, cos, atan2

import gym
import numpy as np

import premaidai_gym

TWO_PI = 2 * pi


class _WalkPhaseGenerator:
    def __init__(self, period, each_stop_period):
        self._period = period
        self._each_stop_period = each_stop_period
        self._move_period = self._period - 2 * self._each_stop_period

    def update(self, normalized_elapsed):
        half_stop_duration = 0.5 * self._each_stop_period
        half_period = 0.5 * self._period
        if 0 < normalized_elapsed <= half_stop_duration:
            phase = 0
        elif (half_stop_duration < normalized_elapsed
              <= half_period - half_stop_duration):
            phase = (normalized_elapsed - half_stop_duration) / self._move_period
        elif (half_period - half_stop_duration < normalized_elapsed
              <= half_period + half_stop_duration):
            phase = 0.5
        elif (half_period + half_stop_duration < normalized_elapsed
              <= self._period - half_stop_duration):
            phase = (normalized_elapsed
                     - 1.5 * self._each_stop_period) / self._move_period
        else:
            phase = 1.0
        return phase


class _BasicWalkController:
    def __init__(self, env: gym.Env, period):
        self._env = env
        self._home_pose = np.full(env.action_space.shape[0], 0.)
        self._home_pose[13] = radians(60)  # right arm
        self._home_pose[18] = radians(-60)  # left arm
        self._period = period
        self._stride_phase_generator = _WalkPhaseGenerator(
            period, each_stop_period=0.15)
        self._bend_phase_generator = _WalkPhaseGenerator(
            period, each_stop_period=0.1)
        self._dt = env.unwrapped.scene.dt
        self._walk_started_at = None

    @property
    def _elapsed(self):
        return self._env.unwrapped.frame * self._dt

    @property
    def _walk_elapsed(self):
        return (self._elapsed - self._walk_started_at
                if self._walk_started_at else 0)

    def step(self, obs):
        normalized_elapsed = self._elapsed % self._period
        phase = normalized_elapsed / self._period
        if phase >= 0.75 and not self._walk_started_at:
            self._walk_started_at = self._elapsed

        roll_wave = radians(5) * sin(TWO_PI * phase)
        if self._walk_started_at:
            phase_stride = self._stride_phase_generator.update(normalized_elapsed)
            stride_wave = radians(10) * cos(TWO_PI * phase_stride)
            phase_bend = self._bend_phase_generator.update(normalized_elapsed)
            bend_wave = radians(20) * sin(TWO_PI * phase_bend)
            if 0 < normalized_elapsed < self._period * 0.5:
                bend_wave_r, bend_wave_l = -bend_wave, 0
            else:
                bend_wave_r, bend_wave_l = 0, bend_wave
        else:
            stride_wave, bend_wave, bend_wave_r, bend_wave_l = 0, 0, 0, 0

        # move legs
        theta_hip_r = -roll_wave
        theta_ankle_r = roll_wave
        r_theta_hip_p = bend_wave_r + stride_wave
        r_theta_knee_p = -2 * bend_wave_r
        r_theta_ankle_p = bend_wave_r - stride_wave
        l_theta_hip_p = bend_wave_l - stride_wave
        l_theta_knee_p = -2 * bend_wave_l
        l_theta_ankle_p = bend_wave_l + stride_wave

        # move arms
        r_theta_sh_p = -2 * stride_wave
        l_theta_sh_p = 2 * stride_wave

        # walking direction control
        cos_yaw, sin_yaw = obs[52], obs[53]
        theta_hip_yaw = 0.3 * bend_wave * atan2(sin_yaw, cos_yaw)

        # roll stabilization
        roll_speed = obs[54]
        theta_ankle_r += 0.3 * roll_speed

        action = np.zeros_like(self._home_pose)
        action[0] = theta_hip_yaw
        action[1] = theta_hip_r
        action[2] = r_theta_hip_p
        action[3] = r_theta_knee_p
        action[4] = r_theta_ankle_p
        action[5] = theta_ankle_r

        action[6] = -theta_hip_yaw
        action[7] = theta_hip_r
        action[8] = l_theta_hip_p
        action[9] = l_theta_knee_p
        action[10] = l_theta_ankle_p
        action[11] = theta_ankle_r

        action[12] = r_theta_sh_p
        action[17] = l_theta_sh_p

        action += self._home_pose
        return action


def main():

    env = gym.make('RoboschoolPremaidAIWalker-v0')

    epi_num = 100
    epis = []
    for epi_i in range(epi_num):
        obs = env.reset()
        walk_controller = _BasicWalkController(env, period=0.91)
        dt = walk_controller._dt
        steps_per_iter = int(10 / dt)  # record 10 sec step
        step = 0
        epi = {'obs': [], 'acs': [], 'rews': [], 'dones': []}
        print(f'Start recording episode for {steps_per_iter}.')
        for step in range(steps_per_iter):
            # elapsed = extra['elapsed']
            action = walk_controller.step(obs)
            obs, reward, done, extra = env.step(action)
            epi['obs'].append(obs)
            epi['acs'].append(action)
            epi['rews'].append(reward)
            epi['dones'].append(int(done))
            # env.render()
            if done:
                break
        print(f'Done episode: {epi_i}, end at step: {step}. Will record result.')
        ep = {k: np.array(v, dtype=np.float32) for k, v in epi.items()}
        epis.append(ep)
        with open('data/expert_epis/RoboschoolPremaidAIWalker-v0_100epis.pkl',
                  'wb') as f:
            pickle.dump(epis, f)


if __name__ == '__main__':
    main()
