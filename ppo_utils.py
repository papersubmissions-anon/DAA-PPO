import torch
import torch.nn as nn
import numpy as np
from torch.distributions.normal import Normal
import dmc2gym
import copy


def make_env(args, seed, idx, capture_video, run_name, env):
    def thunk(env):
        env.seed(args.seed)
        #env = gym.wrappers.RecordEpisodeStatistics(env)
        #if capture_video:
        #    if idx == 0:
        #        env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        #env = gym.wrappers.ClipAction(env)
        #env = gym.wrappers.NormalizeObservation(env)
        #env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        #env = gym.wrappers.NormalizeReward(env)
        #env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        #env.action_space.seed(seed)
        #env.observation_space.seed(seed)
        return env
    return thunk

class reward_normalizer(nn.Module):
    
    def __init__(self, capacity, device):
        super().__init__()
        self.rewards = []
        self.device = device
        self.capacity = capacity
        
    def add(self, reward):
        self.rewards.append(reward)
        if len(self.rewards) > self.capacity:
            self.rewards.pop(0)
        
    def get_mean_std(self):
        rews = torch.tensor(self.rewards).unsqueeze(1).float().to(self.device)
        ex = rews.mean()
        std = rews.std()
        return ex, std
    
    def forward(self, rewards):
        ex, std = self.get_mean_std()
        rewards = (rewards - ex) / (std + 1e-8)
        return rewards

def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)

class Agent(nn.Module):
    def __init__(self, envs, args):
        super(Agent, self).__init__()
        self.critic = nn.Sequential(
            nn.Linear(np.array(envs.observation_space.shape).prod(), args.hidden_dim), nn.ReLU(),
            nn.Linear(args.hidden_dim, args.hidden_dim), nn.ReLU(),
            nn.Linear(args.hidden_dim, 1)
        )
        self.actor_mean = nn.Sequential(
            nn.Linear(np.array(envs.observation_space.shape).prod(), args.hidden_dim), nn.ReLU(),
            nn.Linear(args.hidden_dim, args.hidden_dim), nn.ReLU(),
            nn.Linear(args.hidden_dim, np.array(envs.action_space.shape).prod())
        )
        
        self.record = []
        self.args = args 
        self.apply(weight_init)
        self.simple_logstd = args.simple_logstd
        
        if args.simple_logstd:
            self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.action_space.shape)))
        else:
            self.actor_logstd = nn.Sequential(
            nn.Linear(np.array(envs.observation_space.shape).prod(), args.hidden_dim), nn.ReLU(),
            nn.Linear(args.hidden_dim, np.array(envs.action_space.shape).prod())
        )
            self.apply(weight_init)
        
    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = torch.tanh(self.actor_mean(x))
        if self.simple_logstd:
            action_logstd = self.actor_logstd.expand_as(action_mean)
        else:
            action_logstd = self.actor_logstd(x)
        action_std = torch.exp(action_logstd).clip(min=1e-8)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)

class transition_net(nn.Module):
    def __init__(self, envs, hidden_dim):
        super().__init__()
        
        self.transition = nn.Sequential(
            nn.Linear(np.array(envs.observation_space.shape).prod() + np.array(envs.action_space.shape).prod(), hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, np.array(envs.observation_space.shape).prod()))
        self.apply(weight_init)
        
    def forward(self, state, action):
        state_action = torch.cat([state, action], dim=1)
        return self.transition(state_action)
    
class reward_net(nn.Module):
    def __init__(self, envs, hidden_dim):
        super().__init__()
        
        self.reward = nn.Sequential(
            nn.Linear(np.array(envs.observation_space.shape).prod() + np.array(envs.action_space.shape).prod(), hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1))
        self.apply(weight_init)
        
    def forward(self, state, action):
        state_action = torch.cat([state, action], dim=1)
        return self.reward(state_action)
    
def evaluate(args, agent, num_episodes, device):
    args.seed += 1
    envs_copy = dmc2gym.make(domain_name=args.gym_id,task_name=args.task_name, seed=args.seed, visualize_reward=False, from_pixels=False, frame_skip=args.action_repeat)
    rews = 0
    for i in range(num_episodes):
        obs = torch.tensor(envs_copy.reset()).float().to(device)
        done = False
        while not done:
            with torch.no_grad():
                action, _, _, _ = agent.get_action_and_value(obs.unsqueeze(0))
            obs, reward, done, _ = envs_copy.step(action.cpu().clip(min=-1, max=1).squeeze(0).numpy())
            obs = torch.tensor(obs).float().to(device)
            rews += reward
    agent.record.append(rews/num_episodes)
    return rews/num_episodes

def generate_trajectory(states, horizon, t_net, r_net, agent, device):
    with torch.no_grad():
        states_ = torch.zeros(states.size(0), states.size(1), horizon).float().to(device)
        final_state_values_ = torch.zeros(states.size(0), 1).float().to(device)
        rewards_ = torch.zeros(states.size(0), horizon).float().to(device)
        values_ = torch.zeros(states.size(0), horizon).float().to(device)
        log_probs_ = torch.zeros(states.size(0), horizon).float().to(device)
        actions, logprobs, _, value = agent.get_action_and_value(states)
        actions_ = torch.zeros(states.size(0), actions.size(1), horizon).float().to(device)
        for i in range(horizon):
            states_[:,:,i] = states
            actions_[:,:,i] = actions
            log_probs_[:,i] = logprobs
            values_[:,i] = value.squeeze()
            new_states = t_net(states, actions.clip(min=-1.0, max=1.0))
            rewards_[:,i] = r_net(states, actions.clip(min=-1.0, max=1.0)).squeeze()
            actions, logprobs, _, value = agent.get_action_and_value(new_states)
            states = new_states
        final_state_values_[:] = value
    return states_, actions_, rewards_, final_state_values_.squeeze(), log_probs_, values_

def view_chunk(tensor, chunks, dim=0):
    assert tensor.shape[dim] % chunks == 0
    if dim < 0:  # Support negative indexing
        dim = len(tensor.shape) + dim
    cur_shape = tensor.shape
    new_shape = cur_shape[:dim] + (chunks, tensor.shape[dim] // chunks) + cur_shape[dim + 1:]
    return tensor.reshape(*new_shape).transpose(0,1)

def calculate_advantage(args, final_state_values_, rewards_, values_, device, normalizer=None):
    if args.normalize_rewards:
        rewards_ = normalizer.forward(rewards_)
    next_value = final_state_values_
    if args.gae:
        advantages = torch.zeros_like(rewards_).to(device)
        lastgaelam = 0
        for idx in reversed(range(rewards_.size(1))):
            if idx == rewards_.size(1) - 1:
                nextvalues = next_value
            else:
                nextvalues = values_[:, idx+1]
            delta = rewards_[:, idx] + args.gamma * nextvalues - values_[:, idx]
            advantages[:, idx] = lastgaelam = delta + args.gamma * args.gae_lambda * lastgaelam
        returns = advantages + values_
    if args.gae is False:
        returns = torch.zeros_like(rewards_).to(device)
        for idx in reversed(range(rewards_.size(1))):
            if idx == rewards_.size(1) - 1:
                next_return = next_value
            else:
                next_return = returns[:, idx+1]
            returns[:, idx] = rewards_[:, idx] + args.gamma * next_return
        advantages = returns - values_
    return advantages.detach(), returns.detach()

class ExperienceBuffer(object):
    
    def __init__(self, capacity, env, device):
        super().__init__()
        self.states = torch.zeros(capacity, np.array(env.observation_space.shape).prod())
        self.actions = torch.zeros(capacity, np.array(env.action_space.shape).prod())
        self.rewards = torch.zeros(capacity, 1)
        self.next_states = torch.zeros(capacity, np.array(env.observation_space.shape).prod())
        self.capacity = capacity
        self.full = False
        self.idx = 0
        self.device = device
        
    def add(self, state, action, reward, next_state):
        self.states[self.idx] = state
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.next_states[self.idx] = next_state
        self.idx += 1
        if self.idx >= self.capacity:
            self.full = True
            self.idx = 0
            
    def sample(self, batch_size):
        if self.full:
            idx = np.random.randint(0, self.capacity if self.full else self.idx, size=batch_size)
        else:
            if self.idx < batch_size:
                idx = np.random.randint(0, self.capacity if self.full else self.idx, size=self.idx)
            else:
                idx = np.random.randint(0, self.capacity if self.full else self.idx, size=batch_size)
        states = self.states[idx]
        actions = self.actions[idx]
        rewards = self.rewards[idx]
        new_states = self.next_states[idx]
        return states.to(self.device), actions.to(self.device), rewards.to(self.device), new_states.to(self.device)

def get_q_value(qpos, qvel, state, args, agent, device, normalizer):
    env_copy = dmc2gym.make(domain_name=args.gym_id,task_name=args.task_name, seed=args.seed, visualize_reward=False, from_pixels=False, frame_skip=args.action_repeat)
    s = []
    r = []
    v = []
    _ = torch.tensor(env_copy.reset()).float().to(device)
    qpqv = np.concatenate((qpos,qvel))
    with env_copy.env.physics.reset_context():
      env_copy.env.physics.set_state(qpqv)
    s.append(state)
    with torch.no_grad():
        action, logprob, _, val = agent.get_action_and_value(state.unsqueeze(0))
    obs, reward, done, _ = env_copy.step(action.cpu().clip(min=-1, max=1).squeeze(0).numpy())
    obs = torch.tensor(obs).float().to(device)
    r.append(torch.tensor(reward).float().to(device))
    v.append(val)
    for i in range(args.horizon-1):
        s.append(obs)
        with torch.no_grad():
            a, _, _, _ = agent.get_action_and_value(obs.unsqueeze(0))
        obs, reward, done, _ = env_copy.step(a.cpu().clip(min=-1, max=1).squeeze(0).numpy())
        obs = torch.tensor(obs).float().to(device)
        r.append(torch.tensor(reward).float().to(device))
        v.append(val)
        if done:
            break
    final_state = obs
    s = torch.stack(s)
    r = torch.stack(r)
    v = torch.stack(v).squeeze()
    if args.normalize_rewards:
        r = normalizer.forward(r)
    with torch.no_grad():
        next_value = agent.get_value(final_state.unsqueeze(0)).reshape(1, -1)
        if args.gae:
            advantages = torch.zeros_like(r).to(device)
            lastgaelam = 0
            for t in reversed(range(args.horizon)):
                if t == args.horizon - 1:
                    nextvalues = next_value
                else:
                    nextvalues = v[t + 1]
                delta = r[t] + args.gamma * nextvalues - v[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * lastgaelam
            returns = advantages + v
        else:
            returns = torch.zeros_like(r).to(device)
            for t in reversed(range(args.horizon)):
                if t == args.horizon - 1:
                    next_return = next_value
                else:
                    next_return = returns[t + 1]
                returns[t] = r[t] + args.gamma * next_return
            advantages = returns - v
    return action, logprob, advantages[0]

def get_q_value2(qpos, qvel, state, args, agent, device):
    env_copy = dmc2gym.make(domain_name=args.gym_id,task_name=args.task_name, seed=args.seed, visualize_reward=False, from_pixels=False, frame_skip=args.action_repeat)
    s = []
    r = []
    v = []
    _ = torch.tensor(env_copy.reset()).float().to(device)
    qpqv = np.concatenate((qpos,qvel))
    with env_copy.env.physics.reset_context():
      env_copy.env.physics.set_state(qpqv)
    s.append(state)
    with torch.no_grad():
        action, logprob, _, val = agent.get_action_and_value(state.unsqueeze(0))
    obs, reward, done, _ = env_copy.step(action.cpu().clip(min=-1, max=1).squeeze(0).numpy())
    obs = torch.tensor(obs).float().to(device)
    r.append(torch.tensor(reward).float().to(device))
    v.append(val)
    for i in range(50):
        s.append(obs)
        with torch.no_grad():
            a, _, _, _ = agent.get_action_and_value(obs.unsqueeze(0))
        obs, reward, done, _ = env_copy.step(a.cpu().clip(min=-1, max=1).squeeze(0).numpy())
        obs = torch.tensor(obs).float().to(device)
        r.append(torch.tensor(reward).float().to(device))
        v.append(val)
        if done:
            break
    final_state = obs
    s = torch.stack(s)
    r = torch.stack(r)
    v = torch.stack(v).squeeze()
    with torch.no_grad():
        if done:
            next_value = 0
        else:
            next_value = agent.get_value(final_state.unsqueeze(0)).reshape(1, -1)
        returns = torch.zeros_like(r).to(device)
        for t in reversed(range(51)):
            if t == 50:
                next_return = next_value
            else:
                next_return = returns[t + 1]
            returns[t] = r[t] + args.gamma * next_return
    return action, logprob, returns[0]