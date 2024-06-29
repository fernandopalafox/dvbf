import gymnasium as gym
from gym import spaces
import numpy as np
import cv2


class ResizePixelObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        gym.utils.RecordConstructorArgs.__init__(self, shape=shape)
        gym.ObservationWrapper.__init__(self, env)

        if isinstance(shape, int):
            shape = (shape, shape)
        self.shape = tuple(shape)
        spaces_dict = self.observation_space.spaces
        old_pixel_space = spaces_dict["pixels"]
        spaces_dict["pixels"] = spaces.Box(
            low=0,
            high=255,
            shape=self.shape + (old_pixel_space.shape[2],),
            dtype=np.uint8,
        )
        self.observation_space = spaces_dict

    def observation(self, observation):
        obs_dict = dict(observation)
        pixels = obs_dict["pixels"]
        resized_pixels = cv2.resize(
            pixels,
            self.shape[::-1],
            interpolation=cv2.INTER_AREA,
        )
        grayscale_pixels = cv2.cvtColor(resized_pixels, cv2.COLOR_RGB2GRAY)
        if grayscale_pixels.ndim == 2:
            grayscale_pixels = np.expand_dims(grayscale_pixels, -1)
        obs_dict["pixels"] = grayscale_pixels
        return obs_dict
