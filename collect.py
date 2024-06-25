import gymnasium as gym
from gymnasium.wrappers import PixelObservationWrapper
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
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
        self.observation_space = spaces.Dict(spaces_dict)

    def observation(self, observation):
        obs_dict = dict(observation)
        pixels = obs_dict["pixels"]
        resized_pixels = cv2.resize(
            pixels,
            self.shape[::-1],
            interpolation=cv2.INTER_AREA,
        )
        if resized_pixels.ndim == 2:
            resized_pixels = np.expand_dims(resized_pixels, -1)
        obs_dict["pixels"] = resized_pixels
        return obs_dict


# Test the pixel observation wrapper
env = ResizePixelObservation(
    PixelObservationWrapper(
        gym.make("Pendulum-v1", g=9.81, render_mode="rgb_array")
    ),
    16,
)
observation, info = env.reset()
print(observation["pixels"].shape, info)
env.close()

plt.figure()
plt.imshow(observation["pixels"])
plt.show()
