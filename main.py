import argparse
import io
import time
from datetime import datetime
from pathlib import Path
from typing import Optional
from src.td3_agent_CarRacing import CarRacingTD3Agent

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from racecar_gym.env import RaceEnv
import gymnasium as gym



# ****************************
# accu_time needed to be added
# ****************************

# class RandomAgent:
#     def __init__(self, config):
#         self.scenario = config["scenario"]
#         self.env = Env(scenario=self.scenario,
#                   render_mode='rgb_array_birds_eye',
#                   reset_when_collision=True if 'austria' in self.scenario else False,
#                   output_freq=config["output_freq"])

        
#         self.action_space = gym.spaces.Box(low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float32)
#         # self.env = RaceEnv(scenario=self.scenario,
#         #           render_mode='rgb_array_birds_eye',
#         #           reset_when_collision=True if 'austria' in self.scenario else False)
        
#     def act(self, observation):
#         return self.action_space.sample()
    
#     def train(self):
#         obs, info = self.env.reset()
#         while True:

#             # Decide an action based on the observation (Replace this with your RL agent logic)
#             # action_to_take = agent.decide_agent_actions(obs)  # Replace with actual action
#             action_to_take = self.act(obs)
            
#             # Send an action and receive new observation, reward, and done status
#             obs, reward, terminal, trunc, info = self.env.set_action(action_to_take)


#             if terminal:
#                 print('Episode finished.')
#                 return


if __name__ == '__main__':
    # train env

    config = {
        "gpu": True,
        "training_steps": 1e8,
        "gamma": 0.99,
        "tau": 0.005,
        "batch_size": 128,
        # "warmup_steps": 1000,
        "warmup_steps": 500,
        "total_episode": 100000,
        "lra": 4.5e-5,
        "lrc": 4.5e-5,
        "replay_buffer_capacity": 5000,
        # "logdir": 'log/CarRacing/td3_test_variance_3/',
        "update_freq": 3,
        "eval_interval": 10,
        "eval_episode": 1,
        # "scenario" : "austria_competition",
        "scenario" : "circle_cw_competition_collisionStop",
        "output_freq" : 5,
        "spec_name": 'test',
        "verbosity" : False,
    }
    agent = CarRacingTD3Agent(config)
    
    # rand_agent = RandomAgent(config)
    # agent.train()
    agent.train('/work/u7938613/mathew/TWCC/RL/rl_final_car/results/test_frame_no_sky/weights/model_46716_0.pth')
    # agent.load_and_evaluate('/work/u7938613/mathew/TWCC/RL/rl_final_car/results/test_frame_no_sky/weights/model_46716_0.pth')