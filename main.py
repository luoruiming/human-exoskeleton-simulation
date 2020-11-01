"""
    Author: Ruiming Luo
    Date: 2020.1.15
    Adapted from:
     [1] Udacity DRL Nanodegree program's DDPG implementation. (programming style and project framework)
     https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-pendulum
     [2] Baidu 1st place's solution for NeurIPS2019-Learn-to-Move-Challenge. (algorithm and hyper-parameters)
     https://github.com/PaddlePaddle/PARL/tree/develop/examples/NeurIPS2019-Learn-to-Move-Challenge
"""

import numpy as np
from osim.env import L2M2019Env
import torch
import scipy
import argparse
from collections import deque
from ddpg_agent import Agent, ReplayBuffer
from env_wrapper import FrameSkip, ActionScale, OfficialObs, RewardShaping
from opensim_util import *

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

# load the environment
env = L2M2019Env(difficulty=1, visualize=False)
env.change_model(model='2D')
max_time_limit = env.time_limit
print('max_time_limit:', max_time_limit)

# apply RL tricks
# env = RewardShaping(env)  # reward shaping
env = OfficialObs(env, max_time_limit)  # reshape observation

# size of each action
action_size = env.get_action_space_size()
print('Size of each action:', action_size)

# reset the environment
obs = env.reset(project=False, obs_as_dict=False, init_pose=INIT_POSE)
state_size = obs.size
print('Size of each observation', state_size)

##############################################################################################

parser = argparse.ArgumentParser()
parser.add_argument('--train', dest='train', action='store_true', help='train agent locally')
parser.add_argument('--test', dest='test', action='store_true', help='test agent locally')
args = parser.parse_args()

agent = Agent(state_size=state_size, action_size=action_size, random_seed=0)


def load_memory(agent, memory_file):
    f = open(memory_file, 'r')
    memory_dict = eval(f.read())
    states = memory_dict['state']
    actions = memory_dict['action']
    rewards = memory_dict['reward']
    next_states = memory_dict['next_state']
    assert len(states) == len(actions) == len(rewards) == len(next_states)
    for i in range(len(states)):
        agent.memory.add(states[i], actions[i], rewards[i], next_states[i])
    f.close()


def collect_frames(n_frame, ref_traj):
    # collect n_frame states by sampling trajectories in the ref_traj using random policy
    count = 0
    memory = None
    while True:
        start_id = np.random.randint(ref_traj.shape[0])
        init_pose = np.array([
            ref_traj[start_id][state_idx['pelvis_speed']],     # forward speed
            .5,                                                # rightward speed
            ref_traj[start_id][state_idx['pelvis_ty']],        # pelvis height
            ref_traj[start_id][state_idx['pelvis_tilt']],      # trunk lean
            ref_traj[start_id][state_idx['hip_adduction_r']],  # [right] hip adduct
            ref_traj[start_id][state_idx['hip_flexion_r']],    # hip flex
            ref_traj[start_id][state_idx['knee_angle_r']],     # knee extend
            ref_traj[start_id][state_idx['ankle_angle_r']],    # ankle flex
            ref_traj[start_id][state_idx['hip_adduction_l']],  # [left] hip adduct
            ref_traj[start_id][state_idx['hip_flexion_l']],    # hip flex
            ref_traj[start_id][state_idx['knee_angle_l']],     # knee extend
            ref_traj[start_id][state_idx['ankle_angle_l']]])   # ankle flex

        env.reset(project=False, obs_as_dict=False, init_pose=INIT_POSE)
        while True:
            action = np.random.rand(action_size)
            action = np.clip(action, 0, 1)
            state, _, done, _ = env.step(action, project=False, obs_as_dict=False)

            memory = state if memory is None else np.vstack((memory, state))
            count += 1
            if count % (n_frame / 10) == 0:
                print('Collected {} frames'.format(count))
            if count == n_frame:
                np.save('obs_scaler.npy', memory)
                print('Successfully save ' + str(n_frame) + ' frames to obs_scaler.npy!')
                return memory
            if done:
                break


def ddpg(n_episodes=3000, max_t=max_time_limit, solved_score=200.0, print_every=50, ref_traj=None):
    scores_window = deque(maxlen=print_every)
    scores_res = []
    highest_score = -float('inf')
    for i_episode in range(1, n_episodes+1):
        state = env.reset(project=False, obs_as_dict=False, init_pose=INIT_POSE)     # reset the environment
        score = 0                                                                    # initialize the score
        agent.reset()
        for t in range(max_t):
            action = agent.act(state, add_noise=True)                                # select an action
            action = np.squeeze(action)
            state_, r, done, _ = env.step(action, project=False, obs_as_dict=True)   # send action to environment
            shaped_r = r if ref_traj is None else reward_shaping(state_, ref_traj[t])
            state_ = env.get_observation_from_dict(state_)
            agent.step(state, action, shaped_r, state_)                              # accumulate an experiment and learn a step
            score += shaped_r                                                        # update the score
            state = state_                                                           # roll over states to next time step
            if done:                                                                 # exit loop if episode finished
                print('\tt:', t)
                break
        scores_window.append(score)
        scores_res.append(score)
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")

        if highest_score < np.mean(scores_window):
            highest_score = np.mean(scores_window)
            torch.save(agent.actor_local.state_dict(), '../saved_weights/checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), '../saved_weights/checkpoint_critic.pth')
            print('\nNetwork saved!')

        if i_episode % print_every == 0:
            np.save('scores_res.npy', scores_res)
            print('\nscore saved!')
            # print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window) >= solved_score:
            print('\nEnvironment solved in {} episodes!\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            break
    return scores_res


if args.train:
    # env = FrameSkip(env)  # frame-skip

    # # load the saved weights
    # agent.restore('saved_weights')

    # print('Successfully load memory!')
    ref_traj = np.load('../reference_trajectory.npy')
    print('Successfully load reference trajectory!')

    # load pre-collected memory
    # collect_frames(10000, ref_traj)

    # train the agent with DDPG and save the result
    train_scores = ddpg(max_t=max_time_limit, ref_traj=ref_traj)
    np.save("train_scores.npy", train_scores)

    # plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(train_scores)), train_scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.savefig('pic/reward_curve.png')
    plt.show()

elif args.test:
    # load the saved weights
    agent.actor_local.load_state_dict(torch.load('../saved_weights/checkpoint_actor.pth', map_location='cpu'))
    agent.critic_local.load_state_dict(torch.load('../saved_weights/checkpoint_critic.pth', map_location='cpu'))
    print('Successfully load network weights!')

    state = env.reset(project=False, obs_as_dict=False, init_pose=INIT_POSE)
    scores = 0

    activation = np.array([])
    # muscle order in action:
    #    HAB, HAD, HFL, GLU, HAM, RF, VAS, BFSH, GAS, SOL, TA 

    for i in range(max_time_limit):
        action = agent.act(state, add_noise=False)
        action = np.squeeze(action)
        action_plot = (action + 1) / 2
        activation = np.append(activation, action_plot[1]) # 1:ADD, 2:HFL
        state_, r, done, _ = env.step(action, project=False, obs_as_dict=False)
        scores += r
        state = state_
        if done:
            print('Total score: {:.2f}\tStep:{}'.format(scores, i))
            break

    plt.figure()
    activation_ = scipy.signal.savgol_filter(activation,11,3)
    plt.plot(activation_)
    plt.title('ADD')
    plt.xlabel('Timestep')
    plt.ylabel('Muscle Activation')
    plt.show()

env.close()
