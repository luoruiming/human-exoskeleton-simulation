from osim.env import L2M2019Env
from osim.control.osim_loco_reflex_song2019 import OsimReflexCtrl
import numpy as np
import matplotlib.pyplot as plt
from env_wrapper import FrameSkip, ActionScale, OfficialObs, RewardShaping

mode = '2D'
difficulty = 1  # 0 for collecting data; 1 or 2 for round 1
visualize = True
seed = None
sim_dt = 0.01
sim_t = 100
timstep_limit = int(round(sim_t/sim_dt))
MEMORY_SIZE = int(1e4)

INIT_POSE = np.array([
    1.699999999999999956e+00,    # forward speed
    .5,                          # rightward speed
    9.023245653983965608e-01,    # pelvis height
    2.012303881285582852e-01,    # trunk lean
    0 * np.pi / 180,             # [right] hip adduct
    -6.952390849304798115e-01,   # hip flex
    -3.231075259785813891e-01,   # knee extend
    1.709011708233401095e-01,    # ankle flex
    0 * np.pi / 180,             # [left] hip adduct
    -5.282323914341899296e-02,   # hip flex
    -8.041966456860847323e-01,   # knee extend
    -1.745329251994329478e-01])  # ankle flex

if mode is '2D':
    params = np.loadtxt('params_2D_init.txt')
elif mode is '3D':
    params = np.loadtxt('params_3D_init.txt')

locoCtrl = OsimReflexCtrl(mode=mode, dt=sim_dt)
env = L2M2019Env(visualize=visualize, seed=seed, difficulty=difficulty)
env.change_model(model=mode, difficulty=difficulty, seed=seed)
max_time_limit = env.time_limit
print('max_time_limit:', max_time_limit)

# apply RL tricks
# env = RewardShaping(env)
env = OfficialObs(env, max_time_limit)  # reshape observation


def collect_memory(file_name):
    count = 0
    memory = {'state': [], 'action': [], 'reward': [], 'next_state': []}
    obs_dict = env.reset(project=False, obs_as_dict=True, init_pose=INIT_POSE)
    total_reward = 0
    t = 0
    while True:  # run an episode
        count += 1
        t += sim_dt

        locoCtrl.set_control_params(params)
        action = locoCtrl.update(obs_dict)
        state = env.get_observation_from_dict(obs_dict)
        obs_dict, reward, done, info = env.step(action, project=False, obs_as_dict=True)
        next_state = env.get_observation_from_dict(obs_dict)
        memory['state'].append(state.tolist())
        memory['action'].append(action)
        memory['reward'].append(reward)
        memory['next_state'].append(next_state.tolist())

        total_reward += reward
        if done:
            break
        if count % 1000 == 0:
            print('{} steps finished!'.format(count))
    print('score={:.4f} step={}, time={}sec'.format(total_reward, count, t))

    assert len(memory['state']) == MEMORY_SIZE and count == MEMORY_SIZE

    f = open(file_name, 'w')
    f.write(str(memory))
    f.close()


def test_property():
    total_reward, t, count = 0, 0, 0
    v = np.array([])
    obs_dict = env.reset(project=False, obs_as_dict=True, init_pose=INIT_POSE)
    while True:  # run an episode
        count += 1
        t += sim_dt

        locoCtrl.set_control_params(params)
        action = locoCtrl.update(obs_dict)
        obs_dict, reward, done, info = env.step(action, project=False, obs_as_dict=True)
        total_reward += reward

        v = np.append(v, obs_dict['l_leg']['joint']['hip_abd'])

        # print('l_grf:', obs_dict['l_leg']['ground_reaction_forces'])
        # print('r_grf:', obs_dict['r_leg']['ground_reaction_forces'])
        # input()

        if done:
            break
    print('score={:.4f} step={}, time={}sec'.format(total_reward, count, t))
    plt.plot(v)
    plt.show()


# collect_memory('replay_buffer')
test_property()
