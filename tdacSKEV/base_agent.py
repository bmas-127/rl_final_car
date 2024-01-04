import torch
import torch.nn as nn
import numpy as np
import os
import time
from collections import deque
from torch.utils.tensorboard import SummaryWriter
from src.replay_buffer.replay_buffer import ReplayMemory
from abc import ABC, abstractmethod
import gymnasium as gym
import cv2

class GaussianNoise:
    def __init__(self, dim, mu=None, std=None):
        self.mu = mu if mu else np.zeros(dim)
        self.std = np.ones(dim) * std if std else np.ones(dim) * .1

    def reset(self):
        pass

    def generate(self):
        return np.random.normal(self.mu, self.std)

class OUNoiseGenerator:
    def __init__(self, mean, std_dev, theta=0.3, dt=5e-2):
        self.theta = theta
        self.dt = dt
        self.mean = mean
        self.std_dev = std_dev

        self.x = None

        self.reset()

    def reset(self):
        self.x = np.zeros_like(self.mean.shape)

    def generate(self):
        self.x = (self.x
                  + self.theta * (self.mean - self.x) * self.dt
                  + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape))

        return self.x

class TD3BaseAgent(ABC):
    def __init__(self, config):
        self.gpu = config["gpu"]
        self.device = torch.device("cuda" if self.gpu and torch.cuda.is_available() else "cpu")
        # self.device = "cpu"
        self.total_time_step = 0
        self.training_steps = int(config["training_steps"])
        self.batch_size = int(config["batch_size"])
        self.warmup_steps = config["warmup_steps"]
        self.first_steps = config["first_steps"]
        self.second_steps = config["second_steps"]
        self.third_steps = config["third_steps"]
        self.total_episode = config["total_episode"]
        self.eval_interval = config["eval_interval"]
        self.eval_episode = config["eval_episode"]
        self.gamma = config["gamma"]
        self.tau = config["tau"]
        self.update_freq = config["update_freq"]
        self.action_space = gym.spaces.Box(low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float32)
        self.logdir = config["logdir"]
       
        self.num_frames =  int(config["frame_stake_num"])
        self.resized_dim = int(config["resized_dim"])
        self.frame_stack = []
        self.velocity_stack = []
        self.action_stack = []
        
        self.stacked_frames = []
        self.stacked_action = []
        
        self.replay_buffer = ReplayMemory(int(config["replay_buffer_capacity"]))
        self.writer = SummaryWriter(config["logdir"])

    @abstractmethod
    def decide_agent_actions(self, state, sigma=0.0):
        ### TODO ###
        # based on the behavior (actor) network and exploration noise

        return NotImplementedError
    
    
    @abstractmethod
    def masters_agent_actions(self, state, velocity, sigma=0.0):
        ### TODO ###
        # based on the behavior (actor) network and exploration noise

        return NotImplementedError


    def update(self):
        # update the behavior networks
        self.update_behavior_network()
        # update the target networks
        if self.total_time_step % self.update_freq == 0:
            self.update_target_network(self.target_actor_net, self.actor_net, self.tau)
            self.update_target_network(self.target_critic_net1, self.critic_net1, self.tau)
            self.update_target_network(self.target_critic_net2, self.critic_net2, self.tau)

    @abstractmethod
    def update_behavior_network(self):
        # sample a minibatch of transitions
        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size, self.device)

        ### TODO ###
        # calculate the loss and update the behavior network

        return NotImplementedError


    def process_frame_stack(self, new_frame):
        preprocessed_frame = cv2.cvtColor(np.transpose(new_frame, (1, 2, 0)), cv2.COLOR_RGB2GRAY)
        preprocessed_frame = cv2.resize(preprocessed_frame, (self.resized_dim, self.resized_dim))

        self.frame_stack.append(preprocessed_frame)
        self.frame_stack = self.frame_stack[-self.num_frames:]

        stacked_frames = np.stack(self.frame_stack, axis=0)

        return stacked_frames    
    
    def process_velocity_stack(self, new_velocity):
        self.velocity_stack.append(new_velocity)
        self.velocity_stack = self.velocity_stack[-self.num_frames:]

        stacked_v = np.stack(self.velocity_stack, axis=0)

        return stacked_v 
    
    def process_action_stack(self, new_action):
        self.action_stack.append(new_action)
        self.action_stack = self.action_stack[-self.num_frames:]

        stacked_v = np.stack(self.action_stack, axis=0)

        return stacked_v 
    
    @staticmethod
    def update_target_network(target_net, net, tau):
        # update target network by "soft" copying from behavior network
        for target, behavior in zip(target_net.parameters(), net.parameters()):
            target.data.copy_((1 - tau) * target.data + tau * behavior.data)
            
    def init_env(self, obs):
        preprocessed_frame = cv2.cvtColor(np.transpose(obs, (1, 2, 0)), cv2.COLOR_RGB2GRAY)
        preprocessed_frame = cv2.resize(preprocessed_frame, (self.resized_dim, self.resized_dim))
        self.frame_stack = [preprocessed_frame] * self.num_frames
        self.stacked_frames = self.process_frame_stack(obs)
        
        self.action_stack = [[0.0, 0.0]] * self.num_frames
        self.stacked_action = self.process_action_stack([0.0, 0.0])
        
        return np.array([0.0, 0.0])
        
    
    def act(self, obs):
        self.stacked_frames = self.process_frame_stack(obs)
        action = self.decide_agent_actions(self.stacked_frames, self.stacked_action)
        self.stacked_action = self.process_action_stack(action)
        
        return action
    
    def init_env(self, obs):
        preprocessed_frame = cv2.cvtColor(np.transpose(obs, (1, 2, 0)), cv2.COLOR_RGB2GRAY)
        preprocessed_frame = cv2.resize(preprocessed_frame, (self.resized_dim, self.resized_dim))
        self.frame_stack = [preprocessed_frame] * self.num_frames
        self.stacked_frames = self.process_frame_stack(obs)
        
        self.action_stack = [[0.0, 0.0]] * self.num_frames
        self.stacked_action = self.process_action_stack([0.0, 0.0])
        
        return np.array([0.0, 0.0])
        
    
    def act(self, obs):
        self.stacked_frames = self.process_frame_stack(obs)
        action = self.decide_agent_actions(self.stacked_frames, self.stacked_action)
        self.stacked_action = self.process_action_stack(action)
        
        return action
        

    def train(self):
        for episode in range(self.total_episode):
            total_reward = 0
            original_reward = 0
            state, infos = self.env.reset()
            self.noise.reset()

            preprocessed_frame = cv2.cvtColor(np.transpose(state, (1, 2, 0)), cv2.COLOR_RGB2GRAY)
            preprocessed_frame = cv2.resize(preprocessed_frame, (self.resized_dim, self.resized_dim))
            self.frame_stack = [preprocessed_frame] * self.num_frames
            stacked_frames = self.process_frame_stack(state)
            
            self.velocity_stack = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * self.num_frames
            stacked_velocity = self.process_velocity_stack([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            
            self.action_stack = [[0.0, 0.0]] * self.num_frames
            stacked_action = self.process_action_stack([0.0, 0.0])

            demo = 1
                
            for t in range(10000):
                if self.total_time_step < self.warmup_steps:
                    # sigma = max(0.1*(1-episode/self.total_episode), 0.01)
                    # action = self.masters_agent_actions(stacked_frames, stacked_velocity, sigma=sigma)
                    action = [0.5, 0.18] + np.random.normal(loc=0, scale=0.1, size=(2))
                elif self.total_time_step < self.first_steps:
                    if t % 2 == 0:
                        sigma = max(0.1*(1-episode*10/self.total_episode), 0.1)
                        action = self.masters_agent_actions(stacked_frames, stacked_velocity, sigma=sigma)
                        if original_reward > demo:
                            action = [0.5, 0.18] + np.random.normal(loc=0, scale=0.3, size=(2))
                    else:
                        # exploration degree
                        sigma = max(0.1*(1-episode/self.total_episode), 0.01)
                        action = self.decide_agent_actions(stacked_frames, stacked_action, sigma=sigma)
                else:
                    # exploration degree
                    sigma = max(0.1*(1-episode/self.total_episode), 0.01)
                    action = self.decide_agent_actions(stacked_frames, stacked_action, sigma=sigma)
                
                
                # print(stacked_frames.shape)
                next_state, reward, terminates, truncates, info, ori = self.env.step(action)
                next_stacked_frames = self.process_frame_stack(next_state)
                next_stacked_velocity = self.process_velocity_stack(info["velocity"])
                next_stacked_action = self.process_action_stack(action)
                
                
                self.replay_buffer.append(stacked_frames, stacked_action, action, [reward/10], next_stacked_frames, next_stacked_action, [int(terminates)])
                if self.total_time_step >= 200:
                    self.update()

                self.total_time_step += 1
                total_reward += reward
                original_reward += ori
                stacked_frames = next_stacked_frames
                stacked_velocity = next_stacked_velocity
                stacked_action = next_stacked_action
                


                
                if terminates or truncates:
                    self.writer.add_scalar('Train/Episode Reward', total_reward, self.total_time_step)
                    print(
                        'Step: {}\tEpisode: {}\tLength: {:3d}\tReshaped reward: {:.2f}\tOriginal reward: {:.2f}'
                        .format(self.total_time_step, episode+1, t, total_reward, original_reward))

                    break

            if (episode+1) % self.eval_interval == 0:
                # save model checkpoint
                avg_score = self.evaluate()
                self.save(os.path.join(self.writer.log_dir, f"model_{self.total_time_step}_{float(avg_score)}.pth"))
                self.writer.add_scalar('Evaluate/Episode Reward', avg_score, self.total_time_step)

    def evaluate(self):
        print("==============================================")
        print("Evaluating...")
        all_rewards = []
        for episode in range(self.eval_episode):
            total_reward = 0
            state, infos = self.env.reset(True)
            
            preprocessed_frame = cv2.cvtColor(np.transpose(state, (1, 2, 0)), cv2.COLOR_RGB2GRAY)
            preprocessed_frame = cv2.resize(preprocessed_frame, (32, 32))
            self.frame_stack = [preprocessed_frame] * self.num_frames
            stacked_frames = self.process_frame_stack(state)
            
            self.velocity_stack = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * self.num_frames
            stacked_velocity = self.process_velocity_stack([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            
            self.action_stack = [[0.0, 0.0]] * self.num_frames
            stacked_action = self.process_action_stack([0.0, 0.0])
            
            original_reward = 0
            for t in range(10000):
                action = self.decide_agent_actions(stacked_frames, stacked_action)
                next_state, reward, terminates, truncates, info, ori = self.env.step(action)
                total_reward += reward
            
                stacked_frames = self.process_frame_stack(next_state)
                stacked_velocity = self.process_velocity_stack(info["velocity"])
                stacked_action = self.process_action_stack(action)
                
                original_reward += ori
                if terminates or truncates:
                    print(
                        'Episode: {}\tLength: {:3d}\tTotal reward: {:.2f}\tOriginal reward: {:3f}'
                        .format(episode+1, t, total_reward, original_reward))
                    all_rewards.append(original_reward)
                    break

        avg = sum(all_rewards) / self.eval_episode
        print(f"average score: {avg}")
        print("==============================================")
        return avg
    
    def load_and_train(self, load_path):
        self.load(load_path)
        self.train()

    # save model
    def save(self, save_path):
        torch.save(
                {
                    'actor': self.actor_net.state_dict(),
                    'critic1': self.critic_net1.state_dict(),
                    'critic2': self.critic_net2.state_dict(),
                }, save_path)

    # load model
    def load(self, load_path):
        # checkpoint = torch.load(load_path, map_location=torch.device('cpu'))
        checkpoint = torch.load(load_path)
        self.actor_net.load_state_dict(checkpoint['actor'])
        self.critic_net1.load_state_dict(checkpoint['critic1'])
        self.critic_net2.load_state_dict(checkpoint['critic2'])
        
        
    def load_teacher_and_train(self, load_path):
        # checkpoint = torch.load(load_path, map_location=torch.device('cpu'))
        checkpoint = torch.load(load_path)
        self.teacher_net.load_state_dict(checkpoint['actor'])
        self.train()

    # load model weights and evaluate
    def load_and_evaluate(self, load_path):
        self.load(load_path)
        self.evaluate()

