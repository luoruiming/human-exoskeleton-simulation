import numpy as np
import os
import gym


class RewardShaping(gym.Wrapper):
    """ A wrapper for reward shaping, note this wrapper must be the first wrapper. """

    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.vel_penalty_coeff = 1.0
        self.muscle_penalty_coeff = 1.0
        self.penalty_coeff = 3.0

    def reward_shaping(self, state_desc):
        """define your own reward computation function
        Args:
            state_desc(dict): state description for current model
        """
        # Reward for not falling down
        reward = 10.0

        yaw = state_desc['joint_pos']['ground_pelvis'][2]
        current_v_x, current_v_z = rotate_frame(
            state_desc['body_vel']['pelvis'][0],
            state_desc['body_vel']['pelvis'][2], yaw)
        # leftward
        current_v_z = -current_v_z

        # current relative target theta
        target_v_x, target_v_z = state_desc['v_tgt_field'][0][5][5], state_desc['v_tgt_field'][1][5][5]

        vel_penalty = np.linalg.norm([target_v_x - current_v_x, target_v_z - current_v_z])

        muscle_penalty = 0
        for muscle in sorted(state_desc['muscles'].keys()):
            muscle_penalty += np.square(state_desc['muscles'][muscle]['activation'])

        ret_r = reward - (vel_penalty * self.vel_penalty_coeff + muscle_penalty * self.muscle_penalty_coeff) * self.penalty_coeff
        return ret_r

    def step(self, action, **kwargs):
        state_desc, r, done, info = self.env.step(action, **kwargs)
        shaped_r = self.reward_shaping(state_desc)
        return state_desc, shaped_r, done, info

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return obs


class ActionScale(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

    def step(self, action, **kwargs):
        action = (np.copy(action) + 1.0) * 0.5
        action = np.clip(action, 0.0, 1.0)
        return self.env.step(action, **kwargs)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class FrameSkip(gym.Wrapper):
    def __init__(self, env, skip_num=4):
        gym.Wrapper.__init__(self, env)
        self.skip_num = skip_num
        self.frame_count = 0

    def step(self, action, **kwargs):
        r = 0.0
        merge_info = {}
        for k in range(self.skip_num):
            self.frame_count += 1
            obs, reward, done, info = self.env.step(action, **kwargs)
            r += reward
            if done:
                break
        return obs, r, done, merge_info

    def reset(self, **kwargs):
        self.frame_count = 0
        return self.env.reset(**kwargs)


class OfficialObs(gym.Wrapper):
    MASS = 75.16460000000001  # 11.777 + 2 * (9.3014 + 3.7075 + 0.1 + 1.25 + 0.2166) + 34.2366
    G = 9.80665
    LENGTH0 = 1  # leg length
    Fmax = {
        'r_leg': {
            'HAB': 4460.290481,
            'HAD': 3931.8,
            'HFL': 2697.344262,
            'GLU': 3337.583607,
            'HAM': 4105.465574,
            'RF': 2191.74098360656,
            'VAS': 9593.95082,
            'BFSH': 557.11475409836,
            'GAS': 4690.57377,
            'SOL': 7924.996721,
            'TA': 2116.818162
        },
        'l_leg': {
            'HAB': 4460.290481,
            'HAD': 3931.8,
            'HFL': 2697.344262,
            'GLU': 3337.583607,
            'HAM': 4105.465574,
            'RF': 2191.74098360656,
            'VAS': 9593.95082,
            'BFSH': 557.11475409836,
            'GAS': 4690.57377,
            'SOL': 7924.996721,
            'TA': 2116.818162
        }
    }
    lopt = {
        'r_leg': {
            'HAB': 0.0845,
            'HAD': 0.087,
            'HFL': 0.117,
            'GLU': 0.157,
            'HAM': 0.069,
            'RF': 0.076,
            'VAS': 0.099,
            'BFSH': 0.11,
            'GAS': 0.051,
            'SOL': 0.044,
            'TA': 0.068
        },
        'l_leg': {
            'HAB': 0.0845,
            'HAD': 0.087,
            'HFL': 0.117,
            'GLU': 0.157,
            'HAM': 0.069,
            'RF': 0.076,
            'VAS': 0.099,
            'BFSH': 0.11,
            'GAS': 0.051,
            'SOL': 0.044,
            'TA': 0.068
        }
    }

    def __init__(self, env, max_timelimit):
        gym.Wrapper.__init__(self, env)
        self.max_timelimit = max_timelimit
        self.cur_step = 0
        pre_collected = np.load('obs_scaler.npy') if os.path.exists('obs_scaler.npy') else None
        self.mean = None if pre_collected is None else np.mean(pre_collected, axis=0)
        self.std = None if pre_collected is None else np.std(pre_collected, axis=0)

    def step(self, action, **kwargs):
        state_desc, reward, done, info = self.env.step(action, **kwargs)
        self.cur_step += 1
        if 'obs_as_dict' in kwargs and kwargs['obs_as_dict'] is False:
            obs = self.get_observation(state_desc)
        else:
            obs = self.get_observation_dict(state_desc)
        return obs, reward, done, info

    def reset(self, **kwargs):
        state_desc = self.env.reset(**kwargs)
        self.cur_step = 0
        if 'obs_as_dict' in kwargs and kwargs['obs_as_dict'] is False:
            return self.get_observation(state_desc)
        else:
            return self.get_observation_dict(state_desc)

    def get_observation_dict(self, state_desc):
        obs_dict = dict()

        # pelvis state (in local frame)
        obs_dict['pelvis'] = {}
        obs_dict['pelvis']['height'] = state_desc['body_pos']['pelvis'][1]
        obs_dict['pelvis']['pitch'] = -state_desc['joint_pos']['ground_pelvis'][0]  # (+) pitching forward
        obs_dict['pelvis']['roll'] = state_desc['joint_pos']['ground_pelvis'][1]  # (+) rolling around the forward axis (to the right)
        yaw = state_desc['joint_pos']['ground_pelvis'][2]
        dx_local, dy_local = rotate_frame(state_desc['body_vel']['pelvis'][0],
                                          state_desc['body_vel']['pelvis'][2],
                                          yaw)
        dz_local = state_desc['body_vel']['pelvis'][1]
        obs_dict['pelvis']['vel'] = [   dx_local,  # (+) forward
                                        -dy_local,  # (+) leftward
                                        dz_local,  # (+) upward
                                        -state_desc['joint_vel']['ground_pelvis'][0],  # (+) pitch angular velocity
                                        state_desc['joint_vel']['ground_pelvis'][1],   # (+) roll angular velocity
                                        state_desc['joint_vel']['ground_pelvis'][2]]   # (+) yaw angular velocity

        # leg state
        for leg, side in zip(['r_leg', 'l_leg'], ['r', 'l']):
            obs_dict[leg] = {}
            grf = [ f/(self.MASS*self.G) for f in state_desc['forces']['foot_{}'.format(side)][0:3]]  # forces normalized by body weight
            # grt = [ t/(self.MASS*self.G) for t in state_desc['forces']['foot_{}'.format(side)][3:6]]  # torques normalized by body weight
            grfx_local, grfy_local = rotate_frame(-grf[0], -grf[2], yaw)
            if leg == 'r_leg':
                obs_dict[leg]['ground_reaction_forces'] = [ grfx_local,   # (+) forward
                                                            grfy_local,   # (+) lateral (rightward)
                                                            -grf[1]]      # (+) upward
            if leg == 'l_leg':
                obs_dict[leg]['ground_reaction_forces'] = [ grfx_local,   # (+) forward
                                                            -grfy_local,  # (+) lateral (leftward)
                                                            -grf[1]]      # (+) upward

            # joint angles
            obs_dict[leg]['joint'] = {}
            obs_dict[leg]['joint']['hip_abd'] = -state_desc['joint_pos']['hip_{}'.format(side)][1]  # (+) hip abduction
            obs_dict[leg]['joint']['hip'] = -state_desc['joint_pos']['hip_{}'.format(side)][0]  # (+) extension
            obs_dict[leg]['joint']['knee'] = state_desc['joint_pos']['knee_{}'.format(side)][0]  # (+) extension
            obs_dict[leg]['joint']['ankle'] = -state_desc['joint_pos']['ankle_{}'.format(side)][0]  # (+) extension
            # joint angular velocities
            obs_dict[leg]['d_joint'] = {}
            obs_dict[leg]['d_joint']['hip_abd'] = -state_desc['joint_vel']['hip_{}'.format(side)][1]  # (+) hip abduction
            obs_dict[leg]['d_joint']['hip'] = -state_desc['joint_vel']['hip_{}'.format(side)][0]  # (+) extension
            obs_dict[leg]['d_joint']['knee'] = state_desc['joint_vel']['knee_{}'.format(side)][0]  # (+) extension
            obs_dict[leg]['d_joint']['ankle'] = -state_desc['joint_vel']['ankle_{}'.format(side)][0]  # (+) extension

            # muscles
            for MUS, mus in zip(    ['HAB', 'HAD', 'HFL', 'GLU', 'HAM', 'RF', 'VAS', 'BFSH', 'GAS', 'SOL', 'TA'],
                                    ['abd', 'add', 'iliopsoas', 'glut_max', 'hamstrings', 'rect_fem', 'vasti', 'bifemsh', 'gastroc', 'soleus', 'tib_ant']):
                obs_dict[leg][MUS] = {}
                obs_dict[leg][MUS]['f'] = state_desc['muscles']['{}_{}'.format(mus, side)]['fiber_force'] / self.Fmax[leg][MUS]
                obs_dict[leg][MUS]['l'] = state_desc['muscles']['{}_{}'.format(mus, side)]['fiber_length'] / self.lopt[leg][MUS]
                obs_dict[leg][MUS]['v'] = state_desc['muscles']['{}_{}'.format(mus, side)]['fiber_velocity'] / self.lopt[leg][MUS]

        return obs_dict

    def get_observation(self, state_desc):
        obs_dict = self.get_observation_dict(state_desc)

        res = list()

        res.append(obs_dict['pelvis']['height'])
        res.append(obs_dict['pelvis']['pitch'])
        res.append(obs_dict['pelvis']['roll'])
        res.append(obs_dict['pelvis']['vel'][0])
        res.append(obs_dict['pelvis']['vel'][1])
        res.append(obs_dict['pelvis']['vel'][2])
        res.append(obs_dict['pelvis']['vel'][3])
        res.append(obs_dict['pelvis']['vel'][4])
        res.append(obs_dict['pelvis']['vel'][5])

        for leg in ['r_leg', 'l_leg']:
            res += obs_dict[leg]['ground_reaction_forces']
            res.append(obs_dict[leg]['joint']['hip_abd'])
            res.append(obs_dict[leg]['joint']['hip'])
            res.append(obs_dict[leg]['joint']['knee'])
            res.append(obs_dict[leg]['joint']['ankle'])
            res.append(obs_dict[leg]['d_joint']['hip_abd'])
            res.append(obs_dict[leg]['d_joint']['hip'])
            res.append(obs_dict[leg]['d_joint']['knee'])
            res.append(obs_dict[leg]['d_joint']['ankle'])
            for MUS in [
                    'HAB', 'HAD', 'HFL', 'GLU', 'HAM', 'RF', 'VAS', 'BFSH',
                    'GAS', 'SOL', 'TA'
            ]:
                res.append(obs_dict[leg][MUS]['f'])
                res.append(obs_dict[leg][MUS]['l'])
                res.append(obs_dict[leg][MUS]['v'])

        res = np.array(res)

        remaining_time = 1 - self.cur_step / self.max_timelimit
        res = np.append(res, remaining_time)

        # normalize the array
        if self.mean is not None and self.std is not None:
            res = (res - self.mean) / self.std

        return res

    def get_observation_from_dict(self, obs_dict):
        res = list()

        res.append(obs_dict['pelvis']['height'])
        res.append(obs_dict['pelvis']['pitch'])
        res.append(obs_dict['pelvis']['roll'])
        res.append(obs_dict['pelvis']['vel'][0])
        res.append(obs_dict['pelvis']['vel'][1])
        res.append(obs_dict['pelvis']['vel'][2])
        res.append(obs_dict['pelvis']['vel'][3])
        res.append(obs_dict['pelvis']['vel'][4])
        res.append(obs_dict['pelvis']['vel'][5])

        for leg in ['r_leg', 'l_leg']:
            res += obs_dict[leg]['ground_reaction_forces']
            res.append(obs_dict[leg]['joint']['hip_abd'])
            res.append(obs_dict[leg]['joint']['hip'])
            res.append(obs_dict[leg]['joint']['knee'])
            res.append(obs_dict[leg]['joint']['ankle'])
            res.append(obs_dict[leg]['d_joint']['hip_abd'])
            res.append(obs_dict[leg]['d_joint']['hip'])
            res.append(obs_dict[leg]['d_joint']['knee'])
            res.append(obs_dict[leg]['d_joint']['ankle'])
            for MUS in [
                    'HAB', 'HAD', 'HFL', 'GLU', 'HAM', 'RF', 'VAS', 'BFSH',
                    'GAS', 'SOL', 'TA'
            ]:
                res.append(obs_dict[leg][MUS]['f'])
                res.append(obs_dict[leg][MUS]['l'])
                res.append(obs_dict[leg][MUS]['v'])

        res = np.array(res)

        remaining_time = 1 - self.cur_step / self.max_timelimit
        res = np.append(res, remaining_time)

        # normalize the array
        if self.mean is not None and self.std is not None:
            res = (res - self.mean) / self.std

        return res


def rotate_frame(x, y, theta):
    x_rot = np.cos(theta) * x - np.sin(theta) * y
    y_rot = np.sin(theta) * x + np.cos(theta) * y
    return x_rot, y_rot
