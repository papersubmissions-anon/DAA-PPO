import argparse
import os
import random
import time
from distutils.util import strtobool
from ppo_utils import Agent, evaluate, reward_normalizer, ExperienceBuffer, transition_net, reward_net, generate_trajectory, calculate_advantage, view_chunk
import gym
import dmc2gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename('test').rstrip(".py"), help="the name of this experiment")
    parser.add_argument("--gym-id", type=str, default="cheetah", help="the id of the gym environment")
    parser.add_argument("--task_name", type=str, default="run", help="the id of the gym environment")
    parser.add_argument("--action_repeat", type=int, default=4)
    parser.add_argument('--normalize_rewards', default=True, action='store_true')
    parser.add_argument('--simple_logstd', default=True, action='store_true')
    parser.add_argument('--width_sampling', default=True, action='store_true')
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--buffer_size", type=int, default=25000)
    parser.add_argument("--wm_init_steps", type=int, default=10000)
    parser.add_argument("--action_samples", type=int, default=4)
    parser.add_argument("--experiment_repeats", type=int, default=13)
    parser.add_argument("--wm_batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="the learning rate of the optimizer")
    parser.add_argument("--wm_learning_rate", type=float, default=7e-4, help="the learning rate of the optimizer")
    parser.add_argument("--seed", type=int, default=69,
        help="seed of the experiment")
    parser.add_argument("--total_timesteps", type=int, default=1000000,
        help="total timesteps of the experiments")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="ppo-implementation-details",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="weather to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--num-envs", type=int, default=1,
        help="the number of parallel game environments")
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--num-steps", type=int, default=2048,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gae", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Use GAE for advantage computation")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=32,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=10,
        help="the K epochs to update the policy")
    parser.add_argument('--norm_adv', default=False, action='store_true')

    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.0,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args

def main():
    args = parse_args()
    run_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    for experiment in range(args.experiment_repeats):
    # env setup
        envs = dmc2gym.make(domain_name=args.gym_id,task_name=args.task_name, seed=args.seed, visualize_reward=False, from_pixels=False, frame_skip=args.action_repeat)
        agent = Agent(envs, args).to(device)
        t_net = transition_net(envs, args.hidden_dim).to(device)
        r_net = reward_net(envs, args.hidden_dim).to(device)
        optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
        t_optimizer = optim.Adam(t_net.parameters(), lr=args.wm_learning_rate, eps=1e-8)
        r_optimizer = optim.Adam(r_net.parameters(), lr=args.wm_learning_rate, eps=1e-8)
        if args.normalize_rewards:
            normalizer = reward_normalizer(args.buffer_size, device)
        buffer = ExperienceBuffer(args.buffer_size, envs, device)
        wm_activated = False
    
        # ALGO Logic: Storage setup
        obs = torch.zeros((args.num_steps, args.num_envs) + envs.observation_space.shape).to(device)
        actions = torch.zeros((args.num_steps, args.num_envs) + envs.action_space.shape).to(device)
        logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
        rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
        dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
        values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    
        # TRY NOT TO MODIFY: start the game
        global_step = 0
        start_time = time.time()
        next_obs = torch.Tensor(envs.reset()).float().to(device)
        next_done = torch.zeros(args.num_envs).to(device)
        num_updates = args.total_timesteps // args.batch_size
        episode_reward = 0
    
        for update in range(1, num_updates + 1):
            # Annealing the rate if instructed to do so.
            if args.anneal_lr:
                frac = 1.0 - (update - 1.0) / num_updates
                lrnow = frac * args.learning_rate
                optimizer.param_groups[0]["lr"] = lrnow
                wm_lrnow = frac * args.wm_learning_rate
                t_optimizer.param_groups[0]["lr"] = wm_lrnow
                r_optimizer.param_groups[0]["lr"] = wm_lrnow
    
            for step in range(0, args.num_steps):
                global_step += 1 * args.num_envs
                obs[step] = next_obs
                dones[step] = next_done
    
                # ALGO LOGIC: action logic
                with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(next_obs.unsqueeze(0).float())
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob
    
                # TRY NOT TO MODIFY: execute the game and log data.
                state_ = next_obs.clone()
                next_obs, reward, done, info = envs.step(action.cpu().clip(min=-1, max=1).squeeze(0).numpy())
                buffer.add(state_, action.clip(min=-1.0, max=1.0), reward, torch.tensor(next_obs))
                if args.normalize_rewards:
                    normalizer.add(reward)
                episode_reward += reward
                rewards[step] = torch.tensor(reward).to(device).view(-1)
                next_obs = torch.Tensor(next_obs).float().to(device)
                next_done = torch.Tensor(np.array(done)).to(device)
                if done:
                    next_obs = torch.tensor(envs.reset()).float().to(device)
                    #print(f"global_step={global_step}, episodic_return={episode_reward}")
                    episode_reward = 0
    
            # bootstrap value if not done
            if args.normalize_rewards:
                rewards = normalizer.forward(rewards)
            with torch.no_grad():
                next_value = agent.get_value(next_obs.unsqueeze(0)).reshape(1, -1)
                if args.gae:
                    advantages = torch.zeros_like(rewards).to(device)
                    lastgaelam = 0
                    for t in reversed(range(args.num_steps)):
                        if t == args.num_steps - 1:
                            nextnonterminal = 1.0 - next_done
                            nextvalues = next_value
                        else:
                            nextnonterminal = 1.0 - dones[t + 1]
                            nextvalues = values[t + 1]
                        delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                        advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                    returns = advantages + values
                else:
                    returns = torch.zeros_like(rewards).to(device)
                    for t in reversed(range(args.num_steps)):
                        if t == args.num_steps - 1:
                            nextnonterminal = 1.0 - next_done
                            next_return = next_value
                        else:
                            nextnonterminal = 1.0 - dones[t + 1]
                            next_return = returns[t + 1]
                        returns[t] = rewards[t] + args.gamma * nextnonterminal * next_return
                    advantages = returns - values
    
            # flatten the batch
            b_obs = obs.reshape((-1,) + envs.observation_space.shape)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + envs.action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)
    
            # Optimizing the policy and value network
            b_inds = np.arange(args.batch_size)
            clipfracs = []
            if global_step > args.wm_init_steps:
                #optimize WM
                if wm_activated is False:
                    for wm_batch in range(args.wm_init_steps//10):
                        s_, a_, r_, ns_ = buffer.sample(128)
                        ns_pred = t_net(s_, a_)
                        transition_loss = F.mse_loss(ns_pred, ns_)
                        t_optimizer.zero_grad()
                        transition_loss.backward()
                        t_optimizer.step()
                        r_pred = r_net(s_, a_)
                        reward_loss = F.mse_loss(r_pred, r_)
                        r_optimizer.zero_grad()
                        reward_loss.backward()
                        r_optimizer.step()
                    wm_activated = True
                else:
                    for wm_batch in range(args.update_epochs * args.batch_size // args.minibatch_size):
                        s_, a_, r_, ns_ = buffer.sample(args.wm_batch_size)
                        ns_pred = t_net(s_, a_)
                        transition_loss = F.mse_loss(ns_pred, ns_)
                        t_optimizer.zero_grad()
                        transition_loss.backward()
                        t_optimizer.step()
                        r_pred = r_net(s_, a_)
                        reward_loss = F.mse_loss(r_pred, r_)
                        r_optimizer.zero_grad()
                        reward_loss.backward()
                        r_optimizer.step()
                     
            if wm_activated:
                if args.width_sampling:
                    adv = torch.zeros((b_obs.size(0), args.action_samples)).float().to(device)
                    imagined_lps = torch.zeros((b_obs.size(0), args.action_samples)).float().to(device)
                    imagined_as = torch.zeros((b_obs.size(0), b_actions.size(1), args.action_samples)).float().to(device)
                    for sample in range(args.action_samples):
                        s_, a_, r_, fsv_, lp_, v_ = generate_trajectory(b_obs, args.horizon, t_net, r_net, agent, device)
                        advantages_, returns_ = calculate_advantage(args, fsv_, r_, v_, device, normalizer)
                        adv[:, sample] = advantages_[:, 0]
                        imagined_lps[:, sample] = lp_[:, 0]
                        s_ = s_[:,:,0]
                        imagined_as[:,:,sample] = a_[:,:,0]
                    b_logprobs = torch.cat([b_logprobs.unsqueeze(1), imagined_lps], 1)
                    b_advantages = torch.cat([b_advantages.unsqueeze(1), adv], 1)
            for epoch in range(args.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, args.batch_size, args.minibatch_size):
                    end = start + args.minibatch_size
                    mb_inds = b_inds[start:end]
                    if args.width_sampling and wm_activated:
                        batch_obs_ = b_obs[mb_inds]
                        batch_as_ = b_actions[mb_inds]
                        i_ = 0
                        for i_ in range(args.action_samples):
                            batch_obs_ = torch.cat([batch_obs_, s_[mb_inds]], 0)
                            batch_as_ = torch.cat([batch_as_, imagined_as[mb_inds, :, i_]], 0)
                        _, newlogprob, entropy, newvalue, = agent.get_action_and_value(batch_obs_, batch_as_)
                        newvalue = view_chunk(newvalue, args.action_samples+1, 0).squeeze()[:,0]
                        newlogprob = view_chunk(newlogprob.unsqueeze(1), args.action_samples+1, 0).squeeze()
                        logratio = newlogprob - b_logprobs[mb_inds, :]
                        ratio = logratio.exp()
                        mb_advantages = b_advantages[mb_inds, :]
                    else:
                        _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                        logratio = newlogprob - b_logprobs[mb_inds]
                        ratio = logratio.exp()
                        mb_advantages = b_advantages[mb_inds]
    
                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]
    
                    #mb_advantages = b_advantages[mb_inds, :] if args.width_sampling and wm_activated else b_advantages[mb_inds] 
                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
    
                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                    if args.width_sampling and wm_activated:
                        pg_loss = torch.max(pg_loss1, pg_loss2).mean(1).mean()
                    else:
                        pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                    # Value loss
                    newvalue = newvalue.view(-1)
                    if args.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -args.clip_coef,
                            args.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
    
                    entropy_loss = entropy.mean()
                    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
    
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()
    
                if args.target_kl is not None:
                    if approx_kl > args.target_kl:
                        break
    
            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            eval_rew = evaluate(args, agent, 4, device)
            print(f"global_step={global_step}, eval_return={eval_rew}, SPS={int(global_step / (time.time() - start_time))}")
    
            # TRY NOT TO MODIFY: record rewards for plotting purposes
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            writer.add_scalar("losses/explained_variance", explained_var, global_step)
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
    
        envs.close()
        writer.close()
        if experiment == 0:
            global_results = np.zeros((len(agent.record), args.experiment_repeats))
        global_results[:, experiment] = np.array(agent.record)
        name_ = 'results_ppo_w_' + str(args.gym_id) + '_' + str(args.task_name)
        np.save(name_, global_results)
    
main()