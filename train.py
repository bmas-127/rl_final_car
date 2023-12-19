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

#
SERVER_RAISE_EXCEPTION = True

# Time unit: second
MAX_ACCU_TIME = -1

#
env, reward, trunc, info = None, None, None, None
obs: Optional[np.array] = None
sid: Optional[str] = None
output_freq: Optional[int] = None
terminal = False
images = []
step = 0
port: Optional[int] = None
host: Optional[str] = None
scenario: Optional[str] = None

#
accu_time = 0.
last_get_obs_time: Optional[float] = None
end_time: Optional[float] = None


def get_img_views():
    global obs, sid, info, env

    progress = info['progress']
    lap = int(info['lap'])
    score = lap + progress - 1.

    # Get the images
    img1 = env.env.force_render(render_mode='rgb_array_higher_birds_eye', width=540, height=540,
                                position=np.array([4.89, -9.30, -3.42]), fov=120)
    img2 = env.env.force_render(render_mode='rgb_array_birds_eye', width=270, height=270)
    img3 = env.env.force_render(render_mode='rgb_array_follow', width=128, height=128)
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
    draw.text((550, 500), f"ID {sid}", font=font_large, fill=(255, 255, 255))

    img = np.asarray(img)

    return img


def record_video(filename: str):
    global images
    height, width, layers = images[0].shape
    # noinspection PyUnresolvedReferences
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(filename, fourcc, 30, (width, height))
    for image in images:
        video.write(image)
    cv2.destroyAllWindows()
    video.release()


def get_observation():
    """Return the 3x128x128"""
    try:
        global obs, last_get_obs_time

        # Record time
        last_get_obs_time = time.time()

        return obs, terminal
    except Exception as e:
        if SERVER_RAISE_EXCEPTION:
            raise e
        print(e)
        return jsonify({'error': str(e)})


def set_action(action):
    try:
        global obs, reward, terminal, trunc, info, step, output_freq, sid, accu_time

        if terminal:
            return jsonify({'terminal': terminal})

        accu_time += time.time() - last_get_obs_time

        step += 1
        print(action)
        obs, _, terminal, trunc, info = env.step(action)

        progress = info['progress']
        lap = int(info['lap'])
        score = lap + progress - 1.
        env_time = info['time']

        # Print information
        print_info = f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Step: {step} Lap: {info["lap"]}, ' \
                     f'Progress: {info["progress"]:.3f}, ' \
                     f'EnvTime: {info["time"]:.3f} ' \
                     f'AccTime: {accu_time:.3f} '
        if info.get('n_collision') is not None:
            print_info += f'Collision: {info["n_collision"]} '
        if info.get('collision_penalties') is not None:
            print_info += 'CollisionPenalties: '
            for penalty in info['collision_penalties']:
                print_info += f'{penalty:.3f} '

        print(print_info)

        # plt.imshow(obs.transpose(1, 2, 0))
        # plt.show()

        if step % output_freq == 0:
            img = get_img_views()
            # plt.imshow(img)
            # plt.show()
            images.append(img)

        if terminal:
            if round(accu_time) > MAX_ACCU_TIME:
                print(f'[Time Limit Error] Accu time "{accu_time}" violate the limit {MAX_ACCU_TIME} (sec)!')
            cur_time = datetime.now().strftime("%Y%m%d-%H%M%S")
            video_name = f'results/{sid}_{cur_time}_env{env_time:.3f}_acc{round(accu_time)}s_score{score:.4f}.mp4'
            Path(video_name).parent.mkdir(parents=True, exist_ok=True)
            record_video(video_name)
            print(f'============ Terminal ============')
            print(f'Video saved to {video_name}!')
            print(f'===================================')

        return terminal
    except Exception as e:
        if SERVER_RAISE_EXCEPTION:
            raise e
        print(e)
        return jsonify({'error': str(e)})




def get_args():
    global sid, output_freq, port, scenario, host, MAX_ACCU_TIME
    sid = "tester"
    output_freq = 5
    port = 33333
    scenario = "austria_competition"
    if 'austria' in scenario:
        MAX_ACCU_TIME = 900
    else:
        MAX_ACCU_TIME = 600


def train(agent):
    while True:
        # Get the observation
        obs, terminal = get_observation()
        print(obs.shape)
        

        # Decide an action based on the observation (Replace this with your RL agent logic)
        action_to_take = agent.decide_agent_actions(obs)  # Replace with actual action
        print(action_to_take)
        # Send an action and receive new observation, reward, and done status
        terminal = set_action(action_to_take)


        if terminal:
            print('Episode finished.')
            return



if __name__ == '__main__':
    get_args()

    env = RaceEnv(scenario=scenario,
                  render_mode='rgb_array_birds_eye',
                  reset_when_collision=True if 'austria' in scenario else False)
    obs, info = env.reset()
    
    
    class RandomAgent:
        def __init__(self, action_space):
            self.action_space = action_space

        def act(self, observation):
            return self.action_space.sample()


    # Initialize the RL Agent
    import gymnasium as gym

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
    }
    agent = CarRacingTD3Agent(config)
    
    # rand_agent = RandomAgent(
    #     action_space=gym.spaces.Box(low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float32))

    train(agent)