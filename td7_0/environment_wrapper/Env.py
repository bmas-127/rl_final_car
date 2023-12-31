import argparse
import io
import time
from datetime import datetime
from pathlib import Path
from typing import Optional
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from racecar_gym.env import RaceEnv
import gymnasium as gym



# ****************************
# accu_time needed to be added
# ****************************
class Env:
    def __init__(self, scenario, render_mode, reset_when_collision, output_freq, logdir):
        self.MAX_ACCU_TIME = -1
        self.output_freq = output_freq
        self.last_ncollision = 0
        self.total_episode = 0
        self.env = RaceEnv(scenario=scenario,
                  render_mode=render_mode,
                  reset_when_collision=True if 'austria' in scenario else False)
        self.logdir = f"{logdir}/movie"
        
        scenario = scenario
        if 'austria' in scenario:
            self.MAX_ACCU_TIME = 900
        else:
            self.MAX_ACCU_TIME = 600
            
        self.output_freq = output_freq
    
        
        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir)
        
    def reset(self, test=False):
        obs, info = self.env.reset()
        self.images = []
        self.envstep = 0
        self.accu_time = 0
        self.last_ncollision = 0
        self.test = test
        
        if test == False:
            self.total_episode += 1
        
        
        return obs, info
    
    def record_frames(self, obs, info):
        progress = info['progress']
        lap = int(info['lap'])
        score = lap + progress - 1.

        # Get the images
        img1 = self.env.env.force_render(render_mode='rgb_array_higher_birds_eye', width=540, height=540,
                                    position=np.array([4.89, -9.30, -3.42]), fov=120)
        img2 = self.env.env.force_render(render_mode='rgb_array_birds_eye', width=270, height=270)
        img3 = self.env.env.force_render(render_mode='rgb_array_follow', width=128, height=128)
        img4 = (obs.transpose((1, 2, 0))).astype(np.uint8)

        # Combine the images
        img = np.zeros((540, 810, 3), dtype=np.uint8)
        img[0:540, 0:540, :] = img1
        img[:270, 540:810, :] = img2
        img[270 + 10:270 + 128 + 10, 540 + 7:540 + 128 + 7, :] = img3
        img[270 + 10:270 + 128 + 10, 540 + 128 + 14:540 + 128 + 128 + 14, :] = img4

        # Draw the text
        img = Image.fromarray(img)
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype('./racecar_gym/Arial.ttf', 25)
        font_large = ImageFont.truetype('./racecar_gym/Arial.ttf', 35)
        draw.text((5, 5), "Full Map", font=font, fill=(255, 87, 34))
        draw.text((550, 10), "Bird's Eye", font=font, fill=(255, 87, 34))
        draw.text((550, 280), "Follow", font=font, fill=(255, 87, 34))
        draw.text((688, 280), "Obs", font=font, fill=(255, 87, 34))
        draw.text((550, 408), f"Lap {lap}", font=font, fill=(255, 255, 255))
        draw.text((688, 408), f"Prog {progress:.3f}", font=font, fill=(255, 255, 255))
        draw.text((550, 450), f"Score {score:.3f}", font=font_large, fill=(255, 255, 255))
        draw.text((550, 500), f"ID tester", font=font_large, fill=(255, 255, 255))

        img = np.asarray(img)

        self.images.append(img)


    def record_video(self, filename: str):
        height, width, layers = self.images[0].shape
        # noinspection PyUnresolvedReferences
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(filename, fourcc, 30, (width, height))
        for image in self.images:
            video.write(image)
        cv2.destroyAllWindows()
        video.release()
        


    def step(self, action):
        self.envstep += 1
        obs, reward, terminal, trunc, info = self.env.step(action)
        
        original_reward = reward
        
        
        if self.last_ncollision != len(info["collision_penalties"]):
            reward -= (info["collision_penalties"][self.last_ncollision] ** 0.1)
        
        
        self.last_ncollision = len(info["collision_penalties"])
        velocity = np.linalg.norm(info["velocity"]) * 12
        
        # if velocity > 0.5:
        #     reward += 0.5
        # if velocity > 2:
        #     reward += 1
        
        reward += velocity
        
        # self.test = True
        
        if self.logdir.split("_")[0] == "n":
            self.test = False
        
        # Todo
        if self.test == True:
            if self.envstep % self.output_freq == 0:
                self.record_frames(obs, info)
                
            if terminal:
                self.output_video(info)
        
        
        return obs, reward, terminal, trunc, info, original_reward

    def output_info(self, info):        
        progress = info['progress']
        print(info["progress"])
        lap = int(info['lap'])
        score = lap + progress - 1.
        env_time = info['time']

        # Print information
        print_info = f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Step: {self.envstep} Lap: {info["lap"]}, ' \
                        f'Progress: {info["progress"]:.3f}, ' \
                        f'EnvTime: {info["time"]:.3f} ' 
        if info.get('n_collision') is not None:
            print_info += f'Collision: {info["n_collision"]} '
        if info.get('collision_penalties') is not None:
            print_info += 'CollisionPenalties: '
            for penalty in info['collision_penalties']:
                print_info += f'{penalty:.3f} '

        print(print_info)

    
    def output_video(self, info):
        # if round(accu_time) > self.MAX_ACCU_TIME:
        #     print(f'[Time Limit Error] Accu time "{accu_time}" violate the limit {self.MAX_ACCU_TIME} (sec)!')
        cur_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        video_name = f'{self.logdir}/tester_{self.total_episode}_acc{round(0)}s_score{(int(info["lap"])+info["progress"]-1):.4f}.mp4'
        Path(video_name).parent.mkdir(parents=True, exist_ok=True)
        self.record_video(video_name)
        print(f'============ Terminal ============')
        print(f'Video saved to {video_name}!')
        print(f'===================================')


class RandomAgent:
    def __init__(self, config):
        self.scenario = config["scenario"]
        self.env = Env(scenario=self.scenario,
                  render_mode='rgb_array_birds_eye',
                  reset_when_collision=True if 'austria' in self.scenario else False,
                  output_freq=config["output_freq"])

        
        self.action_space = gym.spaces.Box(low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float32)
        # self.env = RaceEnv(scenario=self.scenario,
        #           render_mode='rgb_array_birds_eye',
        #           reset_when_collision=True if 'austria' in self.scenario else False)
        
    def act(self, observation):
        return self.action_space.sample()
    
    def train(self):
        obs, info = self.env.reset()
        while True:

            # Decide an action based on the observation (Replace this with your RL agent logic)
            # action_to_take = agent.decide_agent_actions(obs)  # Replace with actual action
            action_to_take = self.act(obs)
            
            # Send an action and receive new observation, reward, and done status
            obs, reward, terminal, trunc, info = self.env.step(action_to_take)


            if terminal:
                print('Episode finished.')
                return


if __name__ == '__main__':
    # train env

    config = {
        "gpu": True,
        "training_steps": 1e8,
        "gamma": 0.99,
        "tau": 0.005,
        "batch_size": 32,
        "warmup_steps": 1000,
        "total_episode": 100000,
        "lra": 4.5e-5,
        "lrc": 4.5e-5,
        "replay_buffer_capacity": 5000,
        "logdir": 'log/CarRacing/td3_test_variance_3/',
        "update_freq": 3,
        "eval_interval": 10,
        "eval_episode": 10,
        "scenario" : "austria_competition",
        "output_freq" : 5
    }
    # agent = CarRacingTD3Agent(config)
    
    rand_agent = RandomAgent(config)

    rand_agent.train()