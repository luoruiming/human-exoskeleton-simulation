import numpy as np
import random
import copy
from collections import namedtuple, deque
from ddpg_model import Actor, Critic
import torch
import torch.nn.functional as F
import torch.optim as optim

ACT_DIM = 22  # action space
OBS_DIM = 98  # state space

BUFFER_SIZE = int(1e6)
BATCH_SIZE = 128        # batch size for training neural networks
GAMMA = 0.96            # update coefficient in Bellman equation
TAU = 0.001             # soft update coefficient
LR_ACTOR = 3e-5
LR_CRITIC = 3e-5
OU_SIGMA = 0.2          # Ornstein-Uhlenbeck noise parameter
OU_THETA = 0.15         # Ornstein-Uhlenbeck noise parameter
EPSILON = 1.0           # explore->exploit noise process added to act step
EPSILON_DECAY = 1e-6    # decay rate for noise process

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent:
    def __init__(self, state_size=OBS_DIM, action_size=ACT_DIM, random_seed=0):
        """Initialize an Agent object.
        Params
        =====
            state_size (int): dimension of all observation
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.epsilon = EPSILON

        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC)

        self.noise = OUNoise(action_size, random_seed)
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, random_seed)

    def step(self, state, action, reward, next_state):
        """Save an experience in replay buffer and use random samples from buffer to learn."""
        self.memory.add(state, action, reward, next_state)

        if len(self.memory) > BATCH_SIZE:  # begin to learn when replay buffer is full
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)

    def act(self, state, add_noise=True):
        """Return actions for given state as per current policy."""
        state = state[None, :]
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.epsilon * self.noise.sample()
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples
        Q_target = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q_value

        Params
        =====
            experiences (Tuple[torch.Tensor]): tuple of (s,a,r,s',done)
            gamma (float): discount factor
        """
        states, actions, rewards, next_states = experiences

        # ----------------- update critic network weights ---------------- #
        # get predicted next_state actions and Q_values from target models
        actions_next = self.actor_target(next_states)
        q_targets_next = self.critic_target(next_states, actions_next)
        # compute Q targets for current states
        q_targets = rewards + gamma * q_targets_next
        # compute critic loss
        q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(q_expected, q_targets)
        # minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------- update actor network weights ---------------- #
        # compute the loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------- update target networks ------------------ #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)

        # ----------------- update noise -------------------- #
        self.epsilon -= EPSILON_DECAY
        self.noise.reset()

    @staticmethod
    def soft_update(local_model, target_model, tau):
        """Soft update model parameters
        θ_target = τ * θ_local + (1 - τ) * θ_target
        Params
        =====
            local_model: Network weights to be copied from
            target_model: Network weights to be copied to
            tau（float): interpolation parameter
        """
        for local_param, target_param in zip(local_model.parameters(), target_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def restore(self, save_path):
        actor_checkpoint = torch.load(save_path + '/checkpoint_actor.pth')
        self.actor_local.load_state_dict(actor_checkpoint)
        critic_checkpoint = torch.load(save_path + '/checkpoint_critic.pth')
        self.actor_local.load_state_dict(critic_checkpoint)
        print('Successfully load network weights!')


class OUNoise:
    """Ornstein-Uhlenbeck process"""
    def __init__(self, size, seed, mu=0., theta=OU_THETA, sigma=OU_SIGMA):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.state = None
        self.reset()

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for _ in range(len(x))])
        self.state = x + dx
        return self.state

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, seed=0):
        """Initialize a ReplayBuffer object.
        Params
        =====
            buffer_size (int): maximum size of the buffer
            batch_size (int): size of each training batch
        """
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=['state', 'action', 'reward', 'next_state'])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        return states, actions, rewards, next_states

    def __len__(self):
        return len(self.memory)
