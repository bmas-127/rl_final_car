import torch
import torch.nn as nn
import numpy as np
from src.base_agent import TD3BaseAgent
from src.models.CarRacing_model import ActorNetSimple, CriticNetSimple
import random
from src.base_agent import OUNoiseGenerator, GaussianNoise
from racecar_gym.env import RaceEnv
import gymnasium as gym

class CarRacingTD3Agent(TD3BaseAgent):
    def __init__(self, config):
        super(CarRacingTD3Agent, self).__init__(config)
    
        # initialize environment
        self.observation_space = 128
        self.action_space = 2
        self.action_sample = gym.spaces.Box(low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float32)

        
        # behavior network
        #print("self.env.observation_space.shape[0]:", self.env.observation_space.shape[0])
        self.actor_net = ActorNetSimple(self.observation_space, self.action_space, 3)
        self.critic_net1 = CriticNetSimple(self.observation_space, self.action_space, 3)
        self.critic_net2 = CriticNetSimple(self.observation_space, self.action_space, 3)
        
        self.actor_net.to(self.device)
        self.critic_net1.to(self.device)
        self.critic_net2.to(self.device)
        
        # --------------------------
        # target network
        self.target_actor_net = ActorNetSimple(self.observation_space, self.action_space, 3)
        self.target_critic_net1 = CriticNetSimple(self.observation_space, self.action_space, 3)
        self.target_critic_net2 = CriticNetSimple(self.observation_space, self.action_space, 3)
        
        self.target_actor_net.to(self.device)
        self.target_critic_net1.to(self.device)
        self.target_critic_net2.to(self.device)
        self.target_actor_net.load_state_dict(self.actor_net.state_dict())
        self.target_critic_net1.load_state_dict(self.critic_net1.state_dict())
        self.target_critic_net2.load_state_dict(self.critic_net2.state_dict())

        # set optimizer
        self.lra = config["lra"]
        self.lrc = config["lrc"]

        self.actor_opt = torch.optim.Adam(self.actor_net.parameters(), lr=self.lra)
        self.critic_opt1 = torch.optim.Adam(self.critic_net1.parameters(), lr=self.lrc)
        self.critic_opt2 = torch.optim.Adam(self.critic_net2.parameters(), lr=self.lrc)

        # choose Gaussian noise or OU noise

        # noise_mean = np.full(self.env.action_space.shape[0], 0.0, np.float32)
        # noise_std = np.full(self.env.action_space.shape[0], 2.0, np.float32)
        # self.noise = OUNoiseGenerator(noise_mean, noise_std)
        self.noise = GaussianNoise(self.action_space, 0.0, 2.0)


    def decide_agent_actions(self, state, sigma=0.0, brake_rate=0.015): 
        ### TODO ###
        
        with torch.no_grad():   
            state = torch.from_numpy(state).float().to(self.device)
            state = state.unsqueeze(0) 
            print( self.actor_net(state).cpu().data.numpy() )
            action = self.actor_net(state).cpu().data.numpy() + sigma * self.noise.generate()
            
            print(action)
            action[0, 0] = np.clip(action[0, 0], -1.0, 1.0)  
            action[0, 1] = np.clip(action[0, 1], -1.0, 1.0)   

            action = action[0] 
        
        return action

        

    def update_behavior_network(self):
        # sample a minibatch of transitions
        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size, self.device)
        ### TODO ###
        
        #---------------------------------- start
        
        ## Update Critic ##
        # critic loss
        q_value1 = self.critic_net1(state, action)
        q_value2 = self.critic_net2(state, action)

        
        with torch.no_grad():
            # select action a_next from target actor network and add noise for smoothing

            a_next = self.target_actor_net(next_state).cpu().data.numpy() 
            noise = self.noise.generate()  # generate Gaussian noise
            
            # Adding noise and ensuring the action + noise is still within the valid action space
            a_next = a_next + noise 
            a_next[0, 0] = np.clip(a_next[0, 0], -1.0, 1.0)  # First action: -1 to +1
            action[0, 1] = np.clip(action[0, 1], -1.0, 1.0)   
            
            a_next = torch.from_numpy(a_next).float().to(self.device)

            # get next Q values from target critic networks
            q_next1 = self.target_critic_net1(next_state, a_next)
            q_next2 = self.target_critic_net2(next_state, a_next)
       
            # select min q value from q_next1 and q_next2 (double Q learning)
            q_target = reward + self.gamma * torch.min(q_next1, q_next2) * (1 - done)

        # print(1)
        
        # critic loss function
        criterion = nn.MSELoss()
        critic_loss1 = criterion(q_value1, q_target)
        critic_loss2 = criterion(q_value2, q_target)

        # optimize critic
        self.critic_net1.zero_grad()
        critic_loss1.backward()
        self.critic_opt1.step()

        self.critic_net2.zero_grad()
        critic_loss2.backward()
        self.critic_opt2.step()

        ## Delayed Actor(Policy) Updates ##
        if self.total_time_step % self.update_freq == 0:
            ## Update Actor ##
            # actor loss
            action = self.actor_net(state)
            actor_loss = -1 * self.critic_net1(state, action).mean()
            # optimize actor
            self.actor_net.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()
        #----------------------------------  end
        
