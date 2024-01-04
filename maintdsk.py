import argparse
import io
import time
from datetime import datetime
from pathlib import Path
from typing import Optional
from tdacSKAUS.td3_agent_CarRacing import CarRacingTD3Agent

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from racecar_gym.env import RaceEnv
import gymnasium as gym

if __name__ == '__main__':
    # train env

    config = {
        "gpu": True,
        "training_steps": 1e8,
        "gamma": 0.99,
        "tau": 0.005,
        "batch_size": 64,
        "warmup_steps": 80000,
        "first_steps": 60000,
        "second_steps": 150000,
        "third_steps": 200000,
        "update_steps" : 1000,
        "total_episode": 100000,
        "lra": 1e-5,
        "lrc": 1e-5,
        "replay_buffer_capacity": 20000,
        "logdir": 'log/DuckRacing/episode_8w_99r_1e5',
        # "logdir": 'log/CarRacing/austria_competition/',
        "update_freq": 3,
        "eval_interval": 15,
        "eval_episode": 3,
        "scenario" : "austria_competition",
        # "scenario" : "circle_cw_competition_collisionStop", 
        "output_freq" : 5,
        "frame_stake_num" : 16,
        "resized_dim" : 32,
        "update_loop_num" : 10,
        "method" : "episode"
    } 
    
    # method
    # # episode #
    config["warmup_step"] = 80000
    config["method"] = "episode"
    
    # # step #
    # config["warmup_step"] = 60000
    # config["method"] = "step"
    
    # # inter #
    # config["warmup_step"] = 60000
    # config["method"] = "inter"
    
    config["log_dir"] = f"log/FinalDuckRacing/{config['method']}_{config['warmup_steps']//10000}w_{config['gamma']}r_{config['lra']}"
    print(config["log_dir"])
    
    ## AUS
    agent = CarRacingTD3Agent(config)
    agent.load_teacher_and_train("/work/u7938613/mathew/TWCC/RL/rl_final_car_velocity/log/CarRacing/austria_ori_teacherforce2stage_dir0.2_lowvar_keeptraining_1/top30/model_1254446_2.782882315921964.pth")
    