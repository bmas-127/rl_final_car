import argparse
import json
import numpy as np
import requests

import io
import time
from datetime import datetime
from pathlib import Path
from typing import Optional
from td7_0.td3_agent_CarRacing import CarRacingTD3Agent

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from racecar_gym.env import RaceEnv
import gymnasium as gym


def connect(agent, url: str = 'http://localhost:5000'):
    first = True
    while True:
        # Get the observation
        response = requests.get(f'{url}')
        if json.loads(response.text).get('error'):
            print(json.loads(response.text)['error'])
            break
        obs = json.loads(response.text)['observation']
        obs = np.array(obs).astype(np.uint8)
        
        # Decide an action based on the observation (Replace this with your RL agent logic)
        action_to_take = np.array([0.0, 0.0])
        if first == True:
            agent.load("/work/u7938613/mathew/TWCC/RL/rl_final_car_velocity/log/CarRacing/austria_ori_teacherforce2stage_dir0.2_lowvar_keeptraining_1/top30/model_1254446_2.782882315921964.pth")
            action_to_take = agent.init_env(obs)
            first = False
        else:
            action_to_take = agent.act(obs)  # Replace with actual action

        # Send an action and receive new observation, reward, and done status
        response = requests.post(f'{url}', json={'action': action_to_take.tolist()})
        if json.loads(response.text).get('error'):
            print(json.loads(response.text)['error'])
            break

        result = json.loads(response.text)
        terminal = result['terminal']

        if terminal:
            print('Episode finished.')
            return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', type=str, default='http://localhost:7000', help='The url of the server.')
    args = parser.parse_args()



    config = {
        "gpu": True,
        "training_steps": 1e8,
        "gamma": 0.95,
        "tau": 0.005,
        "batch_size": 64,
        "warmup_steps": 40000,
        "first_steps": 80000,
        "second_steps": 150000,
        "third_steps": 200000,
        "update_steps" : 1000,
        "total_episode": 100000,
        "lra": 1e-4,
        "lrc": 1e-4,
        "replay_buffer_capacity": 20000,
        "logdir": 'log/DuckRacing/model_acstack_3step4_8w_0.95r_1e_4_from_scratch_v3/',
        # "logdir": 'log/CarRacing/austria_competition/',
        "update_freq": 3,
        "eval_interval": 30,
        "eval_episode": 3,
        "scenario" : "austria_competition",
        # "scenario" : "circle_cw_competition_collisionStop", 
        "output_freq" : 5,
        "frame_stake_num" : 16,
        "resized_dim" : 32,
        "update_loop_num" : 10
    }
    agent = CarRacingTD3Agent(config)
    
    # rand_agent = RandomAgent(config)

    # agent.train()
    # agent.load_and_train("model_austria.pth")
    # agent.load_and_train("model_static.pth")
    connect(agent, url="http://127.0.0.1:6000")
