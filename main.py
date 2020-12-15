"""
    Author: Ruiming Luo
    Date: 2020.12.15
"""

import numpy as np
import torch
from torch.optim import Adam
import gym
import time
import math
import scipy
import argparse

from osim_env import MyEnv
from env_wrapper import FrameSkip, ActionScale, OfficialObs
import ppo_core
from ppo import PPOBuffer
from utils.logx import EpochLogger
from utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs


ref_traj = np.load('../reference_trajectory.npy')
print('Successfully load reference trajectory!')

# load the environment
env = MyEnv(visualize=False, ref_traj=ref_traj)
max_time_limit = env.time_limit  
print('max_time_limit:', max_time_limit)

env = OfficialObs(env, max_time_limit)  # reshape observation

# size of each action
action_size = env.get_action_space_size()
print('Size of each action:', action_size)

# reset the environment
obs = env.reset(project=False, obs_as_dict=False)
state_size = obs.size
print('Size of each observation', state_size)

##############################################################################################

parser = argparse.ArgumentParser()
parser.add_argument('--train', dest='train', action='store_true', help='train agent locally')
parser.add_argument('--test', dest='test', action='store_true', help='test agent locally')
parser.add_argument('--cpu', dest='cpu', type=int, default=1)
parser.add_argument('--seed', dest='seed', type=int, default=0)
args = parser.parse_args()


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
            0,                                                 # rightward speed
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

        env.reset(project=False, obs_as_dict=False)
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


def ppo(obs_dim, act_dim, actor_critic=ppo_core.MLPActorCritic, ac_kwargs=dict(), seed=0,
        steps_per_epoch=6144, epochs=1, gamma=0.95, clip_ratio=0.2, pi_lr=3e-4,
        vf_lr=1e-3, train_pi_iters=10, train_v_iters=10, lamb=0.95, target_kl=0.01,
        logger_kwargs=dict(), save_epoch_freq=50, anneal_sample=32000000,
        time_limit_begin=0.1, time_limit_end=3.0, sample_freq=100):

    def target_path(total_sample_count):
        t = total_sample_count / anneal_sample
        lerp = 1.0 if t > 1.0 else t
        lerp = math.pow(lerp, 4)
        return ((1 - lerp) * time_limit_begin + lerp * time_limit_end) * sample_freq + 1

    # Special function to avoid certain slowdowns from PyTorch + MPI combo.
    setup_pytorch_for_mpi()

    # Set up logger and save configuration
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    # Random seed
    seed += 10000 * proc_id()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create actor-critic module
    ac = actor_critic(obs_dim, act_dim, **ac_kwargs)

    # Sync params across processes
    sync_params(ac)

    # Count variables
    var_counts = tuple(ppo_core.count_vars(module) for module in [ac.pi, ac.v])
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n' % var_counts)

    # Set up experience buffer
    local_steps_per_epoch = int(steps_per_epoch / num_procs())
    buffer = PPOBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lamb)

    # Set up function for computing PPO policy loss
    def compute_loss_pi(data):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']

        # Policy loss
        pi, logp = ac.pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.lt(1 - clip_ratio) | ratio.gt(1 + clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info

    # Set up function for computing value loss
    def compute_loss_v(data):
        obs, ret = data['obs'], data['ret']
        return ((ac.v(obs) - ret) ** 2).mean()

    # Set up optimizers for policy and value function
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    vf_optimizer = Adam(ac.v.parameters(), lr=vf_lr)

    # Set up model saving
    logger.setup_pytorch_saver(ac)

    def update():
        data = buffer.get()

        pi_l_old, pi_info_old = compute_loss_pi(data)
        pi_l_old = pi_l_old.item()
        v_l_old = compute_loss_v(data).item()

        # Train policy with multiple steps of gradient descent
        for iter in range(train_pi_iters):
            pi_optimizer.zero_grad()
            loss_pi, pi_info = compute_loss_pi(data)
            kl = mpi_avg(pi_info['kl'])
            if kl > 1.5 * target_kl:
                logger.log('Early stopping at step %d due to reaching max kl.' % iter)
                break
            loss_pi.backward()
            mpi_avg_grads(ac.pi)
            pi_optimizer.step()

        logger.store(StopIter=iter)

        # Value function learning
        for iter in range(train_v_iters):
            vf_optimizer.zero_grad()
            loss_v = compute_loss_v(data)
            loss_v.backward()
            mpi_avg_grads(ac.v)
            vf_optimizer.step()

        # Log changes from update
        kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']
        logger.store(LossPi=pi_l_old, LossV=v_l_old,
                     KL=kl, Entropy=ent, ClipFrac=cf,
                     DeltaLossPi=(loss_pi.item() - pi_l_old),
                     DeltaLossV=(loss_v.item() - v_l_old))

    # Prepare for interaction with environment
    start_time = time.time()
    o, ep_ret, ep_len = env.reset(project=False, obs_as_dict=False), 0, 0

    # Main loop: collect experience in env and update/log each epoch
    epoch = 0
    while True:
        for t in range(local_steps_per_epoch):
            a, v, logp = ac.step(torch.as_tensor(o, dtype=torch.float32))

            next_o, r, d, _ = env.step(a, project=False, obs_as_dict=False)
            ep_ret += r
            ep_len += 1

            # save and log
            buffer.store(o, a, r, v, logp)
            logger.store(VVals=v)

            # Update obs
            o = next_o

            timeout = ep_len == target_path(epoch * steps_per_epoch)
            terminal = d or timeout
            epoch_ended = (t == local_steps_per_epoch - 1)

            if terminal or epoch_ended:
                if epoch_ended and not terminal:
                    print('Warning: trajectory cut off by epoch at %d steps.' % ep_len, flush=True)
                # if trajectory didn't reach terminal state, bootstrap value target
                if timeout or epoch_ended:
                     _, v, _ = ac.step(torch.as_tensor(o, dtype=torch.float32))
                else:  # terminal
                    v = 0
                buffer.finish_path(v)
                if terminal:
                    # only save EpRet / EpLen if trajectory finished
                    logger.store(EpRet=ep_ret, EpLen=ep_len)
                o, ep_ret, ep_len = env.reset(project=False, obs_as_dict=False), 0, 0

        # Save model
        if epoch % save_epoch_freq == 0:
            logger.save_state({'env': env}, None)

        # Perform PPO update
        update()

        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('VVals', with_min_and_max=True)
        logger.log_tabular('TotalEnvInteracts', (epoch + 1) * steps_per_epoch)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('LossV', average_only=True)
        logger.log_tabular('DeltaLossPi', average_only=True)
        logger.log_tabular('DeltaLossV', average_only=True)
        logger.log_tabular('Entropy', average_only=True)
        logger.log_tabular('KL', average_only=True)
        logger.log_tabular('ClipFrac', average_only=True)
        logger.log_tabular('StopIter', average_only=True)
        logger.log_tabular('Time', (time.time() - start_time) / 3600)
        logger.dump_tabular()

        epoch += 1


if args.train:
    # env = FrameSkip(env)  # frame-skip

    # # load the saved weights
    # agent.restore('saved_weights')

    # load pre-collected memory
    # collect_frames(10000, ref_traj)

    # train the agent with DDPG and save the result
    # train_scores = ddpg(max_t=max_time_limit, ref_traj=ref_traj)

    mpi_fork(args.cpu)  # run parallel code with mpi
    from utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs('ppo', args.seed)
    ppo(obs_dim=state_size,
        act_dim=action_size,
        actor_critic=ppo_core.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[1024] * 2),
        gamma=0.95,
        lamb=0.95,
        seed=args.seed,
        steps_per_epoch=6144,
        epochs=1,
        time_limit_begin=0.1,
        time_limit_end=2.0,
        sample_freq=100,
        anneal_sample=32000000,
        logger_kwargs=logger_kwargs)

elif args.test:
    # load the saved weights
    agent.actor_local.load_state_dict(torch.load('../saved_weights/checkpoint_actor.pth', map_location='cpu'))
    agent.critic_local.load_state_dict(torch.load('../saved_weights/checkpoint_critic.pth', map_location='cpu'))
    print('Successfully load network weights!')

    state = env.reset(project=False, obs_as_dict=False)
    scores = 0

    activation = np.array([])
    # muscle order in action:
    #    HAB, HAD, HFL, GLU, HAM, RF, VAS, BFSH, GAS, SOL, TA 

    for i in range(max_time_limit):
        action = agent.act(state, add_noise=False)
        action = np.squeeze(action)
        action_plot = (action + 1) / 2
        activation = np.append(activation, action_plot[1])  # 1:ADD, 2:HFL
        state_, r, done, _ = env.step(action, project=False, obs_as_dict=False)
        scores += r
        state = state_
        if done:
            print('Total score: {:.2f}\tStep:{}'.format(scores, i))
            break

    plt.figure()
    activation_ = scipy.signal.savgol_filter(activation, 11, 3)
    plt.plot(activation_)
    plt.title('ADD')
    plt.xlabel('Timestep')
    plt.ylabel('Muscle Activation')
    plt.show()

env.close()
